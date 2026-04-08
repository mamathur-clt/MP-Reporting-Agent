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

import certifi
import numpy as np
import pandas as pd
from databricks import sql as databricks_sql
from dotenv import load_dotenv

from app.config import KPI_FORMULAS_TEXT

load_dotenv(override=True)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

_HOST = os.getenv("DATABRICKS_HOST", "")
_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")

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
                "WITH queries are allowed. Key schemas: energy_prod.energy "
                "(v_sessions, v_carts, v_orders, v_calls, v_orders_gcv), "
                "energy_prod.data_science, lakehouse_production.energy, "
                "energy_prod.energy.rpt_texas_daily_pacing."
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


def _get_sql_connection():
    return databricks_sql.connect(
        server_hostname=_HOST.replace("https://", "").strip("/"),
        http_path=_HTTP_PATH,
        access_token=_TOKEN,
    )


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
