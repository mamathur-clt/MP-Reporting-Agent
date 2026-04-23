"""
Analyst tools for the enhanced chat agent.

Provides DataFrame querying, SQL execution, schema introspection,
and OpenAI function-calling tool definitions so the chat agent can
answer ad-hoc analytical questions against real data.
"""

import json
import os
import re
import builtins as _builtins_module
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd

from app.config import KPI_FORMULAS_TEXT
from app.db import get_connection as _get_sql_connection
from app.narrative import _BUSINESS_CONTEXT

# ---------------------------------------------------------------------------
# OpenAI function-calling tool schemas
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_loaded_data",
            "description": (
                "Execute Python/pandas code against the loaded DataFrames. "
                "Variables available: `sessions` (session-level funnel data, "
                "one row per session), `finance` (finance daily actuals, one "
                "row per date x channel). Also pre-imported: pd, np, date, "
                "timedelta, datetime. You MUST assign your final output to a "
                "variable called `result`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python/pandas code to execute. Must set `result = ...`."
                        ),
                    },
                    "explanation": {
                        "type": "string",
                        "description": "One-line explanation of what this code does.",
                    },
                },
                "required": ["code", "explanation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_databricks_sql",
            "description": (
                "Run a read-only SQL SELECT query against Databricks. Use when "
                "the question requires data outside the loaded date range or "
                "from tables not present in the loaded DataFrames. Only SELECT / "
                "WITH queries are allowed.\n\n"
                "KEY TABLES & TRUST LEVELS:\n"
                "  Tier 1 (FINANCE — source of truth for performance reporting):\n"
                "    - energy_prod.energy.rpt_texas_daily_pacing — paced month projections and Plan\n"
                "    - v_sessions, v_carts, v_orders, v_calls, v_orders_gcv in energy_prod.energy\n"
                "  Tier 2 (SESSION-LEVEL — source of truth for decomposition/drivers):\n"
                "    - energy_prod.data_science.mp_session_level_query\n"
                "  Tier 3 (CHANNEL-SPECIFIC UPSTREAM):\n"
                "    - lakehouse_production.common.gsc_search_analytics_d_5 — total organic (matches GSC dashboard)\n"
                "    - lakehouse_production.common.gsc_search_analytics_d_3 — page-level organic\n"
                "    - lakehouse_production.common.gsc_search_analytics_d_1 — query-level organic (undercounts)\n"
                "    - energy_prod.energy.paidsearch_campaign — Google Ads campaign data\n\n"
                "DATA SOURCE SELECTION:\n"
                "  'How are we pacing?' → rpt_texas_daily_pacing\n"
                "  'What is our revenue?' → finance views (v_sessions/v_orders)\n"
                "  'Why did VC change?' → mp_session_level_query (decomposition)\n"
                "  'What happened to SEO traffic?' → gsc_search_analytics_d_5\n"
                "  'Which page types lost clicks?' → gsc_search_analytics_d_3\n"
                "  'Which campaigns are driving paid?' → paidsearch_campaign"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "One-line explanation of what this query does.",
                    },
                },
                "required": ["query", "explanation"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Schema context builder
# ---------------------------------------------------------------------------


def build_schema_context(
    sessions_df: pd.DataFrame,
    finance_df: pd.DataFrame | None = None,
) -> str:
    """Return a concise schema description of available DataFrames for the LLM."""
    parts: list[str] = []

    def _describe(df: pd.DataFrame, title: str, var: str, date_col: str | None):
        lines = [f"### {title} (variable: `{var}`)"]
        lines.append(f"Shape: {len(df):,} rows x {len(df.columns)} columns")
        if date_col and date_col in df.columns:
            dates = df[date_col].dropna()
            if not dates.empty:
                lines.append(f"Date range: {dates.min()} to {dates.max()}")
        lines.append("Columns:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            if dtype == "object" and nunique <= 30:
                vals = sorted(df[col].dropna().unique().tolist(), key=str)
                lines.append(f"  {col}: {', '.join(str(v) for v in vals)}")
            elif dtype == "object":
                sample = sorted(df[col].dropna().unique().tolist(), key=str)[:10]
                lines.append(
                    f"  {col}: ({nunique} unique) "
                    f"{', '.join(str(v) for v in sample)} ..."
                )
            elif (
                nunique <= 2
                and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
            ):
                lines.append(f"  {col}: binary 0/1 flag")
            elif "date" in dtype or "datetime" in dtype:
                lines.append(f"  {col}: date")
            else:
                lines.append(f"  {col}: numeric ({dtype})")
        return "\n".join(lines)

    parts.append(
        _describe(sessions_df, "Session-Level Data", "sessions", "session_start_date_est")
    )
    if finance_df is not None and not finance_df.empty:
        parts.append(
            _describe(finance_df, "Finance Daily Data", "finance", "TheDate")
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pandas executor (restricted namespace)
# ---------------------------------------------------------------------------

_BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "open",
    "input", "breakpoint", "exit", "quit",
}
_SAFE_BUILTINS = {
    k: v
    for k, v in vars(_builtins_module).items()
    if k not in _BLOCKED_BUILTINS
}

# Replace __import__ with a restricted version: pandas/numpy internals need it,
# but we block system-level modules (os, sys, subprocess, etc.).
_ALLOWED_IMPORT_PREFIXES = {
    "pandas", "numpy", "pytz", "dateutil",
    "datetime", "time", "calendar", "zoneinfo",
    "collections", "math", "functools", "itertools",
    "operator", "decimal", "fractions", "statistics", "re", "json",
    "copy", "typing", "abc", "enum", "dataclasses", "numbers",
    "string", "textwrap", "unicodedata", "warnings", "contextlib",
    "io", "struct", "hashlib", "bisect", "heapq",
}
_REAL_IMPORT = _builtins_module.__import__


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".")[0]
    if top in _ALLOWED_IMPORT_PREFIXES:
        return _REAL_IMPORT(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of '{name}' is not allowed in the analyst sandbox.")


_SAFE_BUILTINS["__import__"] = _restricted_import

_DANGEROUS_IMPORT_RE = re.compile(
    r"\b(?:import\s+(?:os|sys|subprocess|shutil|socket|http|urllib|ctypes|signal)"
    r"|from\s+(?:os|sys|subprocess|shutil|socket|http|urllib|ctypes|signal)\b)"
)

MAX_RESULT_ROWS = 100


def execute_pandas_query(
    code: str,
    sessions_df: pd.DataFrame,
    finance_df: pd.DataFrame | None = None,
) -> tuple[str, object | None]:
    """
    Run pandas code in a restricted namespace.
    Returns ``(text_for_llm, display_object_or_None)``.
    """
    if "__import__" in code or "exec(" in code or "eval(" in code:
        return "Error: Forbidden function call detected.", None
    if _DANGEROUS_IMPORT_RE.search(code):
        return "Error: Importing system modules is not allowed.", None

    namespace: dict = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "date": date,
        "timedelta": timedelta,
        "datetime": datetime,
        "sessions": sessions_df.copy(),
        "finance": finance_df.copy() if finance_df is not None else pd.DataFrame(),
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", None

    result = namespace.get("result")
    if result is None:
        return (
            "Code ran successfully but `result` was not set. "
            "Assign your final output to `result`."
        ), None

    if isinstance(result, pd.DataFrame):
        total = len(result)
        if total > MAX_RESULT_ROWS:
            result = result.head(MAX_RESULT_ROWS)
            header = f"({total:,} rows total, showing first {MAX_RESULT_ROWS})\n"
        else:
            header = ""
        return header + result.to_string(), result

    if isinstance(result, pd.Series):
        df_out = result.reset_index()
        return df_out.to_string(), df_out

    return str(result), None


# ---------------------------------------------------------------------------
# SQL executor (read-only)
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE)\b",
    re.IGNORECASE,
)

MAX_SQL_ROWS = 500


def execute_sql(query: str) -> tuple[str, object | None]:
    """
    Run a read-only SQL query against Databricks.
    Returns ``(text_for_llm, DataFrame_or_None)``.
    """
    stripped = query.strip()
    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return "Error: Only SELECT / WITH queries are allowed.", None
    if _FORBIDDEN_SQL_RE.search(stripped):
        return "Error: DDL / DML statements are not allowed.", None

    try:
        with _get_sql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            cols = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=cols)
        truncated = len(df) > MAX_SQL_ROWS
        if truncated:
            df = df.head(MAX_SQL_ROWS)
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass
        note = f"\n(Showing first {MAX_SQL_ROWS} rows)" if truncated else ""
        return df.to_string() + note, df
    except Exception as e:
        return f"SQL Error: {type(e).__name__}: {e}", None


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def build_analyst_system_prompt(
    base_chat_prompt: str,
    schema_context: str,
    current_filters: str = "",
) -> str:
    """Augment the existing chat system prompt with data-access tool context."""
    return f"""{base_chat_prompt}

--- BUSINESS CONTEXT ---

{_BUSINESS_CONTEXT}

--- DATA ACCESS TOOLS ---

You have tools to query data directly. Use them to answer ad-hoc analytical questions.

KPI FORMULAS (session-level: rate = SUM(numerator) / SUM(denominator); flags are binary 0/1):
{KPI_FORMULAS_TEXT}

{schema_context}

{f"Filters currently applied in the app: {current_filters}" if current_filters else ""}

TOOL USAGE GUIDELINES:
- Prefer `query_loaded_data` when the question can be answered from the loaded DataFrames.
- Use `run_databricks_sql` only when data is outside the loaded date range or requires tables not in the DataFrames.
- In pandas code: assign the final output to `result`. Pre-imported: pd, np, date, timedelta, datetime.
- Dates in the sessions DataFrame are Python `date` objects. Compare with `date(2026, 3, 23)`, NOT strings.
- For rates: `col.sum() / denominator_col.sum()` (both are 0/1 flags).
- If a tool call errors, read the error and try a corrected approach.
- Keep output concise — aggregate or filter to the essentials before returning.

DATA SOURCE SELECTION RULES:
- "How are we pacing?" → use rpt_texas_daily_pacing (Tier 1)
- "What is our revenue?" → finance views (Tier 1)
- "Why did VC change?" → session-level decomposition (Tier 2)
- "How are initiatives performing?" → session-level with initiative flags (Tier 2)
- "What happened to SEO traffic?" → gsc_search_analytics_d_5 for total organic
- "Which page types lost clicks?" → gsc_search_analytics_d_3 for page-level
- "Which queries drive this page type?" → gsc_search_analytics_d_1 (accepting undercount)
- "Which campaigns are driving paid?" → paidsearch_campaign (Tier 3)

RECONCILIATION AWARENESS:
- The `finance` DataFrame (Tier 1) is the official source of truth for performance numbers.
- The `sessions` DataFrame (Tier 2) is session-level data used for WHY questions (decomposition, drivers, initiative impact).
- Sessions should match closely between finance and session-level, but cart/order metrics may differ slightly due to channel attribution timing: finance assigns channel based on the cart/call session's traffic source, while session-level assigns based on the web session that started the cart.
- When presenting both side-by-side, ALWAYS label the source (e.g., "Finance: 12,345 orders; Session-level: 12,280 orders — difference due to channel attribution timing in phone order reclassification").
- NEVER present session-level numbers as if they were the headline metric. Finance is always the headline; session-level explains the "why."
- If the user asks why numbers don't match, explain the channel attribution timing difference.

--- END DATA ACCESS ---"""


# ---------------------------------------------------------------------------
# Tool executor factory
# ---------------------------------------------------------------------------


def make_tool_executor(
    sessions_df: pd.DataFrame,
    finance_df: pd.DataFrame | None = None,
):
    """Return a callable that dispatches tool calls to the right executor."""

    def _execute(fn_name: str, args_json: str) -> tuple[str, object | None, str]:
        """Returns ``(result_text, display_object, explanation)``."""
        args = json.loads(args_json)
        explanation = args.get("explanation", "")

        if fn_name == "query_loaded_data":
            result_str, result_obj = execute_pandas_query(
                args.get("code", ""), sessions_df, finance_df,
            )
        elif fn_name == "run_databricks_sql":
            result_str, result_obj = execute_sql(args.get("query", ""))
        else:
            result_str, result_obj = f"Unknown tool: {fn_name}", None

        return result_str, result_obj, explanation

    return _execute
