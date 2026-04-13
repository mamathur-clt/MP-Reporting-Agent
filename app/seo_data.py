"""
Data layer for SEO organic ranking reports.

Queries `lakehouse_production.common.seo_fact_clarity_keywords_rankings_json`
with dynamic parameters instead of hardcoded CTE-per-page approach.
Reuses the Databricks connection pattern from app.data.
"""

import os
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

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_DOMAINS = [
    "www.choosetexaspower.org",
    "www.chooseenergy.com",
    "www.saveonenergy.com",
]

PAGE_FRIENDLY_NAMES: dict[str, str] = {
    "https://www.choosetexaspower.org/": "CTXP Homepage",
    "https://www.choosetexaspower.org/electricity-rates/": "CTXP Texas",
    "https://www.choosetexaspower.org/electricity-rates/houston/": "CTXP Houston",
    "https://www.choosetexaspower.org/electricity-rates/dallas/": "CTXP Dallas",
    "https://www.choosetexaspower.org/electricity-rates/fort-worth/": "CTXP Fort Worth",
    "https://www.choosetexaspower.org/electricity-providers/": "CTXP Best Providers",
    "https://www.choosetexaspower.org/energy-resources/no-deposit-electricity/": "CTXP No Deposit",
    "https://www.chooseenergy.com/electricity-rates/texas/": "Choose Texas",
    "https://www.chooseenergy.com/electricity-rates/texas/houston/": "Choose Houston",
    "https://www.chooseenergy.com/electricity-rates/texas/dallas/": "Choose Dallas",
    "https://www.saveonenergy.com/electricity-rates/texas/": "SOE Texas",
    "https://www.saveonenergy.com/electricity-rates/texas/houston/": "SOE Houston",
}

POSITION_BUCKETS = ["1", "2", "3-5", "6-10", "11-20", "20+"]

# ── Connection ────────────────────────────────────────────────────────────


def _get_connection():
    return databricks_sql.connect(
        server_hostname=_HOST.replace("https://", "").strip("/"),
        http_path=_HTTP_PATH,
        access_token=_TOKEN,
    )


# ── Query builder ─────────────────────────────────────────────────────────


def _build_seo_query(start_date: str, domains: list[str], device: str) -> str:
    domain_literals = ", ".join(f"'{d}'" for d in domains)

    return f"""
    SELECT
        date,
        organic_results_link,
        organic_results_link_domain AS domain,
        keyword_tracked,
        keyword_tags,
        organic_results_web_rank,
        organic_results_true_rank,
        device,
        location_requested,
        CASE WHEN avg_search_volume = 99 THEN 0
             ELSE avg_search_volume
        END AS search_volume
    FROM lakehouse_production.common.seo_fact_clarity_keywords_rankings_json
    WHERE date >= '{start_date}'
      AND device = '{device}'
      AND organic_results_link NOT ILIKE '%staging%'
      AND organic_results_link NOT ILIKE '%2%'
      AND organic_results_link_domain IN ({domain_literals})
    """


# ── Data fetcher ──────────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner="Querying Databricks for SEO rankings…")
def fetch_seo_rankings(
    start_date: str,
    domains: list[str] | None = None,
    device: str = "mobile",
) -> pd.DataFrame:
    """Fetch keyword-level ranking rows from the Clarity rankings table."""
    if domains is None:
        domains = DEFAULT_DOMAINS

    query = _build_seo_query(start_date, domains, device)

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["organic_results_web_rank", "organic_results_true_rank", "search_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["page_label"] = df["organic_results_link"].map(PAGE_FRIENDLY_NAMES).fillna(df["organic_results_link"])

    df["position_bucket"] = pd.cut(
        df["organic_results_web_rank"],
        bins=[0, 1, 2, 5, 10, 20, float("inf")],
        labels=POSITION_BUCKETS,
        right=True,
    )

    return df


# ── Aggregation helpers (run on the pandas DataFrame) ─────────────────────


def weighted_avg_rank(df: pd.DataFrame) -> float | None:
    """Compute search-volume-weighted average web rank."""
    vol = df["search_volume"]
    rank = df["organic_results_web_rank"]
    total_vol = vol.sum()
    if total_vol == 0:
        return None
    return (rank * vol).sum() / total_vol


def agg_weighted_rank_over_time(
    df: pd.DataFrame,
    group_col: str = "domain",
    freq: str = "W",
) -> pd.DataFrame:
    """
    Return a time-series DataFrame with weighted average rank per group per period.
    `freq` can be "D" (daily) or "W" (weekly, ISO week-ending Sunday).
    """
    tmp = df.copy()
    tmp["period"] = pd.to_datetime(tmp["date"])
    if freq == "W":
        tmp["period"] = tmp["period"].dt.to_period("W").apply(lambda p: p.start_time.date())
    else:
        tmp["period"] = tmp["date"]

    tmp["rank_x_vol"] = tmp["organic_results_web_rank"] * tmp["search_volume"]

    grouped = (
        tmp.groupby([group_col, "period"])
        .agg(
            rank_x_vol=("rank_x_vol", "sum"),
            total_vol=("search_volume", "sum"),
            keyword_count=("keyword_tracked", "nunique"),
        )
        .reset_index()
    )
    grouped["weighted_avg_rank"] = (
        grouped["rank_x_vol"] / grouped["total_vol"].replace(0, float("nan"))
    ).round(2)

    return grouped.drop(columns=["rank_x_vol"]).sort_values(["period", group_col])


def agg_position_distribution(
    df: pd.DataFrame,
    group_col: str = "domain",
    freq: str = "W",
) -> pd.DataFrame:
    """
    Count keywords per position bucket per group per period.
    Returns long-form data suitable for stacked bar / area charts.
    """
    tmp = df.copy()
    tmp["period"] = pd.to_datetime(tmp["date"])
    if freq == "W":
        tmp["period"] = tmp["period"].dt.to_period("W").apply(lambda p: p.start_time.date())
    else:
        tmp["period"] = tmp["date"]

    dist = (
        tmp.groupby([group_col, "period", "position_bucket"], observed=False)
        .agg(keyword_count=("keyword_tracked", "count"))
        .reset_index()
    )
    dist["position_bucket"] = pd.Categorical(
        dist["position_bucket"], categories=POSITION_BUCKETS, ordered=True
    )
    return dist.sort_values(["period", group_col, "position_bucket"])


def agg_page_scorecard(
    df: pd.DataFrame,
    latest_n_days: int = 7,
    prior_n_days: int = 7,
) -> pd.DataFrame:
    """
    Build a page-level scorecard comparing the latest period to the prior period.
    Each period is `n_days` long, ending at the most recent date in the data.
    """
    if df.empty:
        return pd.DataFrame()

    max_date = pd.Timestamp(max(df["date"]))
    curr_start = (max_date - timedelta(days=latest_n_days - 1)).date()
    curr_end = max_date.date()
    prior_end = (max_date - timedelta(days=latest_n_days)).date()
    prior_start = (max_date - timedelta(days=latest_n_days + prior_n_days - 1)).date()

    def _score(sub: pd.DataFrame) -> pd.Series:
        w = weighted_avg_rank(sub)
        return pd.Series({
            "weighted_avg_rank": round(w, 2) if w else None,
            "avg_web_rank": round(sub["organic_results_web_rank"].mean(), 2),
            "keywords_tracked": sub["keyword_tracked"].nunique(),
            "keywords_in_top_10": sub.loc[sub["organic_results_web_rank"] <= 10, "keyword_tracked"].nunique(),
            "total_search_volume": sub["search_volume"].sum(),
        })

    curr = df[(df["date"] >= curr_start) & (df["date"] <= curr_end)]
    prior = df[(df["date"] >= prior_start) & (df["date"] <= prior_end)]

    if curr.empty:
        return pd.DataFrame()

    curr_scores = curr.groupby(["page_label", "domain"]).apply(_score).reset_index()
    prior_scores = prior.groupby(["page_label", "domain"]).apply(_score).reset_index()

    merged = curr_scores.merge(
        prior_scores[["page_label", "domain", "weighted_avg_rank"]].rename(
            columns={"weighted_avg_rank": "prior_weighted_avg_rank"}
        ),
        on=["page_label", "domain"],
        how="left",
    )
    merged["rank_change"] = merged["weighted_avg_rank"] - merged["prior_weighted_avg_rank"]

    return merged.sort_values("weighted_avg_rank")


def default_seo_start_date() -> date:
    """Sensible default: 6 months back."""
    return date.today() - timedelta(days=180)
