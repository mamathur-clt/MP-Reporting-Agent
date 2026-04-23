"""
Data layer: connects to Databricks, runs the session-level funnel query,
returns a pandas DataFrame.  Results are cached per Streamlit session to
avoid re-running the ~700-line query on every interaction.
"""

import os
import ssl
from datetime import date, timedelta

import certifi
import pandas as pd
import streamlit as st
from databricks import sql as databricks_sql
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

_HOST = os.getenv("DATABRICKS_HOST", "")
_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")


def _get_connection():
    return databricks_sql.connect(
        server_hostname=_HOST.replace("https://", "").strip("/"),
        http_path=_HTTP_PATH,
        access_token=_TOKEN,
    )


def _build_query(start_date: str, end_date: str) -> str:
    """Read the session_level_query file and inject date parameters."""
    query_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "queries", "session_level_query"
    )
    with open(query_path) as f:
        raw = f.read()

    # The file starts with %sql — strip that
    raw = raw.replace("%sql", "", 1).strip()

    # Replace the date literals in the cte_variables CTE
    raw = raw.replace(
        "('2026-01-01')::date AS start_date",
        f"('{start_date}')::date AS start_date",
    )
    raw = raw.replace(
        "(CURRENT_DATE)::date AS end_date",
        f"('{end_date}')::date AS end_date",
    )
    return raw


@st.cache_data(ttl=3600, show_spinner="Querying Databricks (this may take a minute)…")
def fetch_session_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Execute the full session-level funnel query and return a DataFrame."""
    query = _build_query(start_date, end_date)
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)

    # Normalise types
    if "session_start_date_est" in df.columns:
        df["session_start_date_est"] = pd.to_datetime(
            df["session_start_date_est"]
        ).dt.date

    # Ensure numeric for all flag columns
    flag_cols = [
        "session", "zip_entry", "has_cart", "cart_order", "cart_page1_done",
        "cart_customer_info_done", "cart_ssn_done", "gross_call", "queue_call",
        "phone_order", "cart_credit_fail", "cart_provider_pass",
        "cart_volt_fail", "cart_qual_fail",
        "is_fmp", "is_lp", "is_grid", "grid_lp",
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # cart_order and phone_order come from SUM() in the SQL and can be >1
    # when a session has multiple cart/phone entries. Cap to 0/1 for rate KPIs.
    for col in ["cart_order", "phone_order"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=1)

    df["total_order"] = df.get("cart_order", 0) + df.get("phone_order", 0)
    df["total_order"] = df["total_order"].clip(upper=1)

    # Clean up dimension values for readability
    _label_cleanups = {
        "mover_switcher": {0: "Unknown", "0": "Unknown", "": "Unknown"},
        "device_type": {0: "Unknown", "0": "Unknown", "": "Unknown"},
        "landing_page_type": {0: "Unknown", "0": "Unknown", "": "Unknown"},
        "first_partner_name": {0: "Unknown", "0": "Unknown", "": "Unknown"},
    }
    for col, mapping in _label_cleanups.items():
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").replace(mapping)

    return df


def default_date_range() -> tuple[date, date]:
    """Return a sensible default: ~8 weeks back from today."""
    today = date.today()
    return today - timedelta(weeks=8), today
