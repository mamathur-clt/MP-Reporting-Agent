"""
Shared Databricks connection helper for the Slack bot.

Consolidates the connection pattern used across app/data.py,
app/analyst_tools.py, app/finance_data.py, and app/seo_data.py
into a single module.
"""

import os
import re
from contextlib import contextmanager

import certifi
import pandas as pd
from databricks import sql as databricks_sql
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

_HOST = os.getenv("DATABRICKS_HOST", "")
_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")

MAX_SQL_ROWS = 500

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE)\b",
    re.IGNORECASE,
)


@contextmanager
def get_connection():
    """Yield a Databricks SQL connection, closing it on exit."""
    conn = databricks_sql.connect(
        server_hostname=_HOST.replace("https://", "").strip("/"),
        http_path=_HTTP_PATH,
        access_token=_TOKEN,
    )
    try:
        yield conn
    finally:
        conn.close()


def execute_readonly_sql(query: str, max_rows: int = MAX_SQL_ROWS) -> tuple[str, pd.DataFrame | None]:
    """
    Run a read-only SQL query against Databricks.

    Returns (text_summary, DataFrame_or_None). Rejects DDL/DML.
    """
    stripped = query.strip()
    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return "Error: Only SELECT / WITH queries are allowed.", None
    if _FORBIDDEN_SQL_RE.search(stripped):
        return "Error: DDL / DML statements are not allowed.", None

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchmany(max_rows + 1)
            cols = [desc[0] for desc in cursor.description]

        df = pd.DataFrame(rows, columns=cols)
        truncated = len(df) > max_rows
        if truncated:
            df = df.head(max_rows)

        for c in df.columns:
            if df[c].dtype == object:
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass

        note = f"\n(Showing first {max_rows} rows)" if truncated else ""
        return df.to_string() + note, df
    except Exception as e:
        return f"SQL Error: {type(e).__name__}: {e}", None
