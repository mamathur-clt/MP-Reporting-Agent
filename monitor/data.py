"""
Partner Health Monitor — data layer.

Reads cart_session_level_query.txt, injects date parameters,
executes against Databricks, and returns a pandas DataFrame.
"""

import os
from datetime import date, datetime, timedelta

import certifi
import pandas as pd
from databricks import sql as databricks_sql

from monitor.config import (
    CART_QUERY_PATH,
    DATABRICKS_HOST,
    DATABRICKS_HTTP_PATH,
    DATABRICKS_TOKEN,
)

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def _get_connection():
    return databricks_sql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", "").strip("/"),
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


def _build_query(start_date: str, end_date: str) -> str:
    """Read cart_session_level_query.txt and inject date parameters."""
    with open(CART_QUERY_PATH) as f:
        raw = f.read()

    raw = raw.replace(
        "('2025-01-01')::date AS start_date",
        f"('{start_date}')::date AS start_date",
    )
    raw = raw.replace(
        "CURRENT_DATE::date AS end_date",
        f"('{end_date}')::date AS end_date",
    )
    return raw


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column types after fetch."""
    if "_date" in df.columns:
        df["_date"] = pd.to_datetime(df["_date"]).dt.date
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)

    flag_cols = [
        "zip_entry", "cart_entry", "cart_order", "credit_fail", "qual_fail",
        "volt_fail", "page_1_completion", "customer_info_completion",
        "ssn_completion", "address_captured", "appointment_selected",
        "clicked_review_page_cta", "pivot_clicked", "pivot_order",
        "pivot_triggered", "rogs_ind", "has_nrg_intent",
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Derived convenience flags used by metrics.py
    if "credit_fail" in df.columns:
        df["credit_pass_flag"] = (1 - df["credit_fail"]).clip(lower=0)
    if "credit_runs" in df.columns:
        df["has_credit_run"] = (pd.to_numeric(df["credit_runs"], errors="coerce").fillna(0) > 0).astype(int)
    if "qual_fail" in df.columns:
        df["qual_pass_flag"] = (1 - df["qual_fail"]).clip(lower=0)
        df["has_qual_result"] = (
            df["midflow_response"].notna() | df["enrollment_response"].notna()
        ).astype(int) if "midflow_response" in df.columns else df["qual_fail"].clip(upper=1)
    if "volt_fail" in df.columns:
        df["volt_pass_flag"] = (1 - df["volt_fail"].fillna(0)).clip(lower=0)
    if "cart_entry" in df.columns:
        df["has_cart_session"] = df["cart_entry"]

    for col in ["cart_order", "ssn_completion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(upper=1).astype(int)

    # Decimal → float for any remaining numeric columns
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                pass

    return df


def fetch_cart_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Execute the cart-session query for a date range and return a DataFrame."""
    query = _build_query(start_date, end_date)
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    return _coerce_types(df)


def fetch_recent_hours(n_hours: int = 1) -> pd.DataFrame:
    """Fetch data for the last *n_hours* hours (plus a buffer day for the query)."""
    end = date.today()
    start = end - timedelta(days=1)
    df = fetch_cart_data(str(start), str(end))
    cutoff = datetime.now() - timedelta(hours=n_hours)
    if "_date" in df.columns and "hour" in df.columns:
        df["_datetime"] = df.apply(
            lambda r: datetime.combine(r["_date"], datetime.min.time()) + timedelta(hours=r["hour"]),
            axis=1,
        )
        df = df[df["_datetime"] >= cutoff].drop(columns=["_datetime"])
    return df


def fetch_baseline(days: int = 14) -> pd.DataFrame:
    """Fetch the rolling baseline window (last *days* days)."""
    end = date.today()
    start = end - timedelta(days=days)
    return fetch_cart_data(str(start), str(end))
