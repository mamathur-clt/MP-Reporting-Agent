"""
Data layer for SEO organic ranking + GSC visibility reports.

Two families of queries live in this module:

1. **SEO Clarity rankings** (the original): keyword-level rank tracking out of
   `lakehouse_production.common.seo_fact_clarity_keywords_rankings_json`.
2. **GSC visibility + landing-page-type analysis** (new in Phase 3): clicks,
   impressions, CTR, and weighted-average rank from the GSC tables plus
   the Organic session funnel by `landing_page_type` from
   `energy_prod.data_science.mp_session_level_query`.

Both reuse the Databricks connection pattern from `app.data`.

GSC table selection (per `docs/prd_v2.md`):
- `gsc_search_analytics_d_5` — site-level totals (no `page`, no `query`);
  matches the GSC dashboard exactly. Use for total organic impressions /
  clicks / CTR / rank trends that need to reconcile with the dashboard.
- `gsc_search_analytics_d_1` — adds `page` AND `query`; the most granular
  table. Use for anything requiring `page` — page-type drill-downs, top
  queries, top declining pages, unmatched URLs.

**d_1 == d_3 at (date, domain, page) grain.** We empirically verified (2026
YTD across all four Texas energy domains, 132,616 page-day combos, zero gaps)
that aggregating d_1 up to (date, domain, page) produces byte-identical
clicks + impressions to d_3 — d_3 is strictly a subset of d_1's columns.
We therefore source every page-level query from d_1 so the tab only has to
reason about two tables (d_5 for site totals, d_1 for anything with `page`)
instead of three. d_5 stays distinct because it retains rows that d_1 drops
due to anonymised long-tail queries, and therefore reconciles with the GSC
dashboard whereas d_1 does not.
"""

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.db import get_connection as _get_connection

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_DOMAINS = [
    "www.choosetexaspower.org",
    "www.chooseenergy.com",
    "www.saveonenergy.com",
]

# GSC tracks bare domains (no `www.`); these are what `d_1/d_3/d_5` return.
GSC_DOMAINS = [
    "choosetexaspower.org",
    "saveonenergy.com",
    "chooseenergy.com",
    "texaselectricrates.com",
]

# GSC domain → session `website` name (used to join GSC to session data).
GSC_DOMAIN_TO_SITE = {
    "choosetexaspower.org": "CTXP",
    "saveonenergy.com": "SOE",
    "chooseenergy.com": "Choose TX",
    "texaselectricrates.com": "TXER",
}

GSC_SITE_TO_DOMAIN = {v: k for k, v in GSC_DOMAIN_TO_SITE.items()}
GSC_SITE_OPTIONS = ["All"] + list(GSC_SITE_TO_DOMAIN.keys())


# ── landing_page_type → bucket taxonomy ───────────────────────────────────
#
# `mp_session_level_query.landing_page_type` returns ~28 distinct values, of
# which the tail (Business, Cart, NTX*, PriceToCompare, GEO, other, …) each
# carry <2% of Organic sessions — too small to steer against individually.
# The agreed exec-view taxonomy (April 2026) compresses them to 6 buckets
# so that week-over-week reports focus on differences that actually matter:
#
#   - Homepage       — brand-driven entry point; kept isolated.
#   - StateGEO       — state-level ranking pages (`StateGEO` + legacy `GEO` stub).
#   - CityGEO        — Tier{1..4}CityGEO rolled up; sub-tier drill-down stays
#                      available via the raw `landing_page_type` column.
#   - Provider       — REP-review templates; distinct intent/funnel from GEO.
#   - PlanType       — plan-feature landing pages AND the Solar content hub
#                      AND the PowerToChoose alt page (per 2026-04 decision).
#   - Informational  — non-commerce / low-VC pages (Resources, Grid,
#                      Utilities, Business, Spanish, Cart, NTX*, etc.).
#
# A 7th value, ``Unmatched``, is emitted by the GSC join for pages that
# don't appear as an Organic landing URL in the session table — this is a
# data-hygiene bucket, not a strategic one, so we keep it separate.
#
# NB: this is the single source of truth for the mapping. Callers in
# `app/tabs/organic_deep_dive.py`, `app/seo_diagnostic.py`, and
# `bot/tools/seo.py` should import ``bucket_for_landing_page_type`` and
# ``LANDING_PAGE_TYPE_BUCKETS`` from here rather than defining their own.
# When a new `landing_page_type` shows up in the session table, add it to
# the mapping below and it will automatically flow through every tab,
# bot tool, and diagnostic that uses this helper.

LANDING_PAGE_TYPE_BUCKETS: tuple[str, ...] = (
    "Homepage",
    "StateGEO",
    "CityGEO",
    "Provider",
    "PlanType",
    "Informational",
)

# Raw landing_page_type → bucket. Keep the keys in sync with what
# `mp_session_level_query` emits — there's a validation query in the
# repo's notebooks that lists the current distinct values (see
# `notebooks/seo_landing_page_type_analysis.ipynb`).
LANDING_PAGE_TYPE_TO_BUCKET: dict[str, str] = {
    "Homepage": "Homepage",
    "StateGEO": "StateGEO",
    "GEO": "StateGEO",
    "Tier1CityGEO": "CityGEO",
    "Tier2CityGEO": "CityGEO",
    "Tier3CityGEO": "CityGEO",
    "Tier4CityGEO": "CityGEO",
    "Provider": "Provider",
    "Solar": "PlanType",
    "Solar_Buyback_Plans": "PlanType",
    "Free_Nights_Weekends_Plans": "PlanType",
    "No_Deposit_Plans": "PlanType",
    "Same_Day_Plans": "PlanType",
    "Term_Length_Plans": "PlanType",
    "PDP_Plans": "PlanType",
    "Other_Plans": "PlanType",
    "PowerToChoose": "PlanType",
    "Resources": "Informational",
    "Utilities": "Informational",
    "NTXUtility": "Informational",
    "NTX": "Informational",
    "Grid": "Informational",
    "Gas": "Informational",
    "Business": "Informational",
    "Spanish": "Informational",
    "Cart": "Informational",
    "PriceToCompare": "Informational",
    "other": "Informational",
}

# "Unmatched" from the GSC join is not in the session table — it's the
# name we assign to GSC URLs that fail the lookup. Keep it isolated.
UNMATCHED_BUCKET = "Unmatched"
UNMATCHED_LANDING_PAGE_TYPE = "Unmatched"


def bucket_for_landing_page_type(lpt: str | None) -> str:
    """Collapse a raw `landing_page_type` value into its exec-view bucket.

    Rules:
      • ``None``/empty → ``Unmatched`` (the GSC-join hygiene bucket).
      • ``Unmatched``  → ``Unmatched`` (passes through).
      • Anything in ``LANDING_PAGE_TYPE_TO_BUCKET`` → that bucket.
      • Anything else (e.g. a brand-new `landing_page_type` value we haven't
        classified yet) → ``Informational``, so it shows up but doesn't
        silently mask traffic from the major buckets.
    """
    if not lpt:
        return UNMATCHED_BUCKET
    if lpt == UNMATCHED_LANDING_PAGE_TYPE:
        return UNMATCHED_BUCKET
    return LANDING_PAGE_TYPE_TO_BUCKET.get(lpt, "Informational")


def _sql_bucket_case_expr(
    landing_page_type_col: str,
    *,
    alias: str | None = "landing_page_type_bucket",
) -> str:
    """SQL fragment that maps a `landing_page_type` column to its bucket.

    Used inside every GSC / session query so bucket assignment happens in
    the warehouse (keeps the query result directly groupable by bucket
    without a pandas merge). Mirrors ``bucket_for_landing_page_type``
    exactly — if the Python mapping is updated, this function regenerates
    the matching SQL automatically.
    """
    lines = [f"CASE"]
    lines.append(
        f"      WHEN {landing_page_type_col} IS NULL "
        f"OR {landing_page_type_col} = '{UNMATCHED_LANDING_PAGE_TYPE}' "
        f"THEN '{UNMATCHED_BUCKET}'"
    )
    for raw, bucket in LANDING_PAGE_TYPE_TO_BUCKET.items():
        # SQL-safe literal: none of our page-type names contain single quotes.
        lines.append(f"      WHEN {landing_page_type_col} = '{raw}' THEN '{bucket}'")
    # Fallback for brand-new values we haven't mapped yet — matches the
    # Python fallback in ``bucket_for_landing_page_type``.
    lines.append("      ELSE 'Informational'")
    lines.append("    END")
    expr = "\n    ".join(lines)
    if alias:
        return f"{expr} AS {alias}"
    return expr

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


@st.cache_data(ttl=7200, show_spinner="Querying Databricks for SEO rankings…")
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


# ═══════════════════════════════════════════════════════════════════════════
# Google Search Console — visibility + landing-page-type analysis
# ═══════════════════════════════════════════════════════════════════════════
#
# All four query helpers below issue a SINGLE Databricks query and return a
# plain pandas DataFrame; aggregation / pivoting for charts happens in the
# Streamlit tab so helpers remain reusable outside the UI (e.g. in the analyst
# chat tool).
# ---------------------------------------------------------------------------


def _gsc_domain_list(domains: list[str] | None = None) -> str:
    """Render a SQL IN-list of GSC domain literals."""
    doms = domains or GSC_DOMAINS
    return ", ".join(f"'{d}'" for d in doms)


def _build_url_to_page_type_cte(cte_start_date: str = "2025-01-01") -> str:
    """
    Deduplicated `landing_page → landing_page_type` lookup, sourced from
    `mp_session_level_query`. Identical to the `url_to_page_type` CTE in
    `notebooks/seo_landing_page_type_analysis.ipynb` — a URL may appear with
    multiple types across sessions, so we pick the most common type per URL
    (by session count).
    """
    return f"""
    url_to_page_type AS (
      SELECT landing_page, landing_page_type
      FROM (
        SELECT
          RTRIM('/', LOWER(first_page_url))  AS landing_page,
          landing_page_type,
          ROW_NUMBER() OVER (
            PARTITION BY RTRIM('/', LOWER(first_page_url))
            ORDER BY SUM(sessions) DESC
          ) AS rn
        FROM energy_prod.data_science.mp_session_level_query
        WHERE _date >= '{cte_start_date}'
          AND landing_page_type IS NOT NULL
        GROUP BY RTRIM('/', LOWER(first_page_url)), landing_page_type
      )
      WHERE rn = 1
    )
    """.strip()


@st.cache_data(ttl=1800, show_spinner="Checking GSC freshness…")
def fetch_gsc_last_available_date(
    domains: list[str] | None = None,
) -> date | None:
    """
    Return the most recent `date` that has at least one row in `d_5` for the
    supplied domains. GSC typically lags real-time by 1–2 days, and the lag
    varies day-to-day — callers should use this to truncate comparison
    windows so both periods cover the same number of *fully-reported* days.
    """
    dom_list = _gsc_domain_list(domains)
    query = f"""
    SELECT MAX(date) AS max_date
    FROM lakehouse_production.common.gsc_search_analytics_d_5
    WHERE domain IN ({dom_list})
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    val = row[0]
    if isinstance(val, date):
        return val
    return pd.to_datetime(val).date()


def align_windows_to_gsc(
    curr_start: date,
    curr_end: date,
    prior_start: date,
    prior_end: date,
    gsc_max_date: date | None,
) -> tuple[date, date, date, date, str]:
    """
    Truncate both the current and prior windows so they cover the same number
    of fully-reported GSC days.

    Rule: current window ends at ``min(curr_end, gsc_max_date)``; both windows
    are then resized to the same day count by clipping the later edge of the
    prior window. If the current window ends up longer than the prior one the
    prior window stays as-is.

    Returns ``(c_start, c_end, p_start, p_end, note)`` where *note* is a human
    readable description of what changed (empty string if nothing changed).
    """
    if gsc_max_date is None:
        return curr_start, curr_end, prior_start, prior_end, ""

    original_curr_end = curr_end
    if curr_end > gsc_max_date:
        curr_end = gsc_max_date

    # Never let an already-degenerate window (curr_start > curr_end) through.
    if curr_end < curr_start:
        # All of the current window is beyond the GSC horizon — clamp both
        # windows to a single day at gsc_max_date / same-offset prior.
        return curr_start, curr_start, prior_start, prior_start, (
            f"GSC only has data through {gsc_max_date}; current window starts after "
            "that, so no comparison is possible yet."
        )

    curr_days = (curr_end - curr_start).days + 1
    prior_days = (prior_end - prior_start).days + 1

    note_parts: list[str] = []
    if curr_end != original_curr_end:
        note_parts.append(
            f"Current window truncated to {curr_end} (GSC's latest fully-reported day)."
        )

    # If prior window is longer than current, shrink it to match (starting at
    # prior_start). We do NOT extend prior beyond its original end.
    if prior_days > curr_days:
        new_prior_end = prior_start + timedelta(days=curr_days - 1)
        note_parts.append(
            f"Prior window shortened to {prior_start} → {new_prior_end} "
            f"({curr_days} days) to match the current window."
        )
        prior_end = new_prior_end
    elif prior_days < curr_days:
        # Prior is shorter than current — shrink current instead so both
        # windows cover an identical number of fully-reported days.
        new_curr_end = curr_start + timedelta(days=prior_days - 1)
        note_parts.append(
            f"Current window further shortened to {curr_start} → {new_curr_end} "
            f"({prior_days} days) to match the prior window."
        )
        curr_end = new_curr_end

    return curr_start, curr_end, prior_start, prior_end, " ".join(note_parts)


@st.cache_data(ttl=7200, show_spinner="Loading GSC P4WA baseline…")
def fetch_gsc_p4wa(
    curr_start: date,
    curr_end: date,
    domains: list[str] | None = None,
) -> dict:
    """
    Prior-4-week-average baseline for the current window.

    Intended for the **WoW** view on the Organic Deep Dive tab. Returns the
    sum-then-divide-by-4 baseline for impressions, clicks, CTR and
    impression-weighted rank over the four week-aligned windows immediately
    preceding ``curr_start``.

    Matches the finance-side P4WA logic (`build_funnel_summary`) in shape —
    we pick the same day-of-week set as the current window so a Mon–Sun
    window is compared to four prior Mon–Sun windows.

    Parameters
    ----------
    curr_start, curr_end : datetime.date
        Inclusive current-window bounds.
    domains : optional list of GSC domains.

    Returns
    -------
    dict with keys ``clicks``, ``impressions``, ``ctr``,
    ``weighted_avg_rank``, ``weeks_used`` (how many of the four weeks had
    data), and ``start_of_range`` / ``end_of_range`` for display.
    """
    if curr_start > curr_end:
        return {}
    curr_days = (curr_end - curr_start).days + 1

    # Four windows, each `curr_days` long, ending the day before `curr_start`.
    windows: list[tuple[date, date]] = []
    ptr_end = curr_start - timedelta(days=1)
    for _ in range(4):
        ptr_start = ptr_end - timedelta(days=curr_days - 1)
        windows.append((ptr_start, ptr_end))
        ptr_end = ptr_start - timedelta(days=1)

    oldest = min(w[0] for w in windows)
    newest = max(w[1] for w in windows)

    dom_list = _gsc_domain_list(domains)
    query = f"""
    SELECT
      date,
      SUM(clicks)                                           AS clicks,
      SUM(impressions)                                      AS impressions,
      SUM(position * impressions)                            AS pos_x_impr
    FROM lakehouse_production.common.gsc_search_analytics_d_5
    WHERE date BETWEEN '{oldest}' AND '{newest}'
      AND domain IN ({dom_list})
    GROUP BY date
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return {
            "clicks": 0.0, "impressions": 0.0, "ctr": None, "weighted_avg_rank": None,
            "weeks_used": 0, "start_of_range": oldest, "end_of_range": newest,
        }
    df["date"] = pd.to_datetime(df["date"]).dt.date

    total_clicks = 0.0
    total_impr = 0.0
    total_pos_x_impr = 0.0
    weeks_with_data = 0
    for w_start, w_end in windows:
        sub = df[(df["date"] >= w_start) & (df["date"] <= w_end)]
        if sub.empty:
            continue
        weeks_with_data += 1
        total_clicks += float(sub["clicks"].sum())
        total_impr += float(sub["impressions"].sum())
        total_pos_x_impr += float(sub["pos_x_impr"].sum())

    if weeks_with_data == 0:
        return {
            "clicks": 0.0, "impressions": 0.0, "ctr": None, "weighted_avg_rank": None,
            "weeks_used": 0, "start_of_range": oldest, "end_of_range": newest,
        }

    # Average across the weeks we actually have.
    avg_clicks = total_clicks / weeks_with_data
    avg_impr = total_impr / weeks_with_data
    ctr = (total_clicks / total_impr) if total_impr else None
    wrank = (total_pos_x_impr / total_impr) if total_impr else None
    return {
        "clicks": avg_clicks,
        "impressions": avg_impr,
        "ctr": ctr,
        "weighted_avg_rank": wrank,
        "weeks_used": weeks_with_data,
        "start_of_range": oldest,
        "end_of_range": newest,
    }


@st.cache_data(ttl=7200, show_spinner="Loading GSC page-type baseline…")
def fetch_gsc_by_page_type_multi_window_avg(
    windows: tuple[tuple[date, date], ...],
    domains: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Averaged ``fetch_gsc_by_page_type`` results across an arbitrary list of
    historical windows — used to build **P4WA** and **P4MA** baselines at the
    landing_page_type × site grain.

    Sourced from `gsc_search_analytics_d_1` (see module docstring for the
    d_1 == d_3 empirical equivalence). Group-by stays at (domain,
    landing_page_type) per window; we do NOT add `query` to the grain.

    Given N windows (e.g. four prior calendar months for MoM MTD, or four
    prior same-weekday 7-day windows for WoW), each window is queried, and
    the output is a per-(site, landing_page_type) average:

        avg_clicks       = total_clicks       / windows_with_data
        avg_impressions  = total_impressions  / windows_with_data
        ctr              = total_clicks       / total_impressions
        weighted_avg_rank = SUM(pos*impr across windows) / total_impressions

    CTR and rank are impression-weighted across *all* windows (not a
    simple mean of per-window CTRs) so the baseline aligns with how the
    current-window CTR is computed.

    Parameters
    ----------
    windows : list of ``(start, end)`` date tuples (inclusive).
    domains : optional GSC domain list.

    Returns
    -------
    pd.DataFrame with columns ``site``, ``domain``, ``landing_page_type``,
    ``clicks``, ``impressions``, ``ctr``, ``weighted_avg_rank``,
    ``windows_used``. If no windows have data, returns an empty frame.
    """
    if not windows:
        return pd.DataFrame()

    dom_list = _gsc_domain_list(domains)
    url_cte = _build_url_to_page_type_cte()

    # Build one big UNION-ALL query labelling each row with its window idx so
    # we can count `windows_used` per (site, landing_page_type).
    window_selects: list[str] = []
    for idx, (w_start, w_end) in enumerate(windows):
        window_selects.append(f"""
        SELECT
          {idx} AS window_idx,
          g.domain,
          COALESCE(u.landing_page_type, 'Unmatched') AS landing_page_type,
          SUM(g.clicks)                               AS clicks,
          SUM(g.impressions)                          AS impressions,
          SUM(g.position * g.impressions)             AS pos_x_impr
        FROM lakehouse_production.common.gsc_search_analytics_d_1 g
        LEFT JOIN url_to_page_type u
          ON RTRIM('/',
               LOWER(
                 CASE WHEN POSITION('#' IN g.page) > 0
                   THEN LEFT(g.page, POSITION('#' IN g.page) - 1)
                   ELSE g.page
                 END
               )
             ) = u.landing_page
        WHERE g.date BETWEEN '{w_start}' AND '{w_end}'
          AND g.domain IN ({dom_list})
        GROUP BY g.domain, COALESCE(u.landing_page_type, 'Unmatched')
        """)
    union_sql = "\nUNION ALL\n".join(window_selects)

    query = f"""
    WITH {url_cte},
    per_window AS (
{union_sql}
    )
    SELECT
      domain,
      landing_page_type,
      COUNT(DISTINCT window_idx) AS windows_used,
      SUM(clicks)                AS total_clicks,
      SUM(impressions)           AS total_impressions,
      SUM(pos_x_impr)            AS total_pos_x_impr
    FROM per_window
    GROUP BY domain, landing_page_type
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in ["windows_used", "total_clicks", "total_impressions", "total_pos_x_impr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["clicks"] = df["total_clicks"] / df["windows_used"].replace(0, pd.NA)
    df["impressions"] = df["total_impressions"] / df["windows_used"].replace(0, pd.NA)
    df["ctr"] = df["total_clicks"] / df["total_impressions"].replace(0, pd.NA)
    df["weighted_avg_rank"] = (
        df["total_pos_x_impr"] / df["total_impressions"].replace(0, pd.NA)
    )
    df["site"] = df["domain"].map(GSC_DOMAIN_TO_SITE).fillna(df["domain"])
    # Bucket label assigned client-side: the bucket is a pure function of
    # `landing_page_type`, and keeping the SQL unchanged here avoids
    # widening every per_window select.
    df["landing_page_type_bucket"] = df["landing_page_type"].map(
        bucket_for_landing_page_type
    )
    return df[[
        "site", "domain", "landing_page_type", "landing_page_type_bucket",
        "clicks", "impressions", "ctr", "weighted_avg_rank",
        "windows_used",
    ]]


def build_p4ma_windows(
    curr_start: date, curr_end: date, months_back: int = 4,
) -> list[tuple[date, date]]:
    """
    Build a list of ``months_back`` prior-month windows, each covering the
    same day-of-month range as the current MTD window.

    Example (current = Apr 1 – Apr 14):
        [(Mar 1, Mar 14), (Feb 1, Feb 14), (Jan 1, Jan 14), (Dec 1, Dec 14)]

    This is the temporal analogue of ``fetch_gsc_p4wa`` used in the MoM MTD
    mode of the Organic Deep Dive tab — "how does this page-type compare to
    the average of the same day-count over the four most-recent months".

    If ``curr_end.day > last_day_of_prior_month`` (happens on the 31st of a
    31-day month when the prior month has 30 days), the prior window is
    clipped to the last valid day of that month. No re-pacing: the user
    should interpret P4MA as "average over the part of each prior month
    that we actually observed".
    """
    import calendar
    if curr_start > curr_end:
        return []
    start_day = curr_start.day
    end_day = curr_end.day

    windows: list[tuple[date, date]] = []
    cursor = date(curr_start.year, curr_start.month, 1)
    for _ in range(months_back):
        if cursor.month == 1:
            prev_y, prev_m = cursor.year - 1, 12
        else:
            prev_y, prev_m = cursor.year, cursor.month - 1
        last_day_prev = calendar.monthrange(prev_y, prev_m)[1]
        w_start = date(prev_y, prev_m, min(start_day, last_day_prev))
        w_end = date(prev_y, prev_m, min(end_day, last_day_prev))
        windows.append((w_start, w_end))
        cursor = date(prev_y, prev_m, 1)
    return windows


def build_p4wa_windows(
    curr_start: date, curr_end: date, weeks_back: int = 4,
) -> list[tuple[date, date]]:
    """
    Build a list of ``weeks_back`` prior same-weekday windows, each covering
    the same length as the current week window.

    Example (current = Mon Apr 11 – Sun Apr 17):
        [(Apr 4 – Apr 10), (Mar 28 – Apr 3), (Mar 21 – Mar 27), (Mar 14 – Mar 20)]

    Exactly parallels the internals of ``fetch_gsc_p4wa`` but exposed as a
    reusable window builder for page-type-level averaging via
    ``fetch_gsc_by_page_type_multi_window_avg``.
    """
    if curr_start > curr_end:
        return []
    window_days = (curr_end - curr_start).days + 1
    windows: list[tuple[date, date]] = []
    ptr_end = curr_start - timedelta(days=1)
    for _ in range(weeks_back):
        ptr_start = ptr_end - timedelta(days=window_days - 1)
        windows.append((ptr_start, ptr_end))
        ptr_end = ptr_start - timedelta(days=1)
    return windows


@st.cache_data(ttl=7200, show_spinner="Computing GSC page-1 churn…")
def fetch_gsc_page1_churn(
    curr_start: date,
    curr_end: date,
    prior_start: date,
    prior_end: date,
    domains: list[str] | None = None,
    min_prior_impressions: int = 50,
) -> dict:
    """
    Share of queries that were ranked page 1 in the prior window (avg rank ≤ 4)
    and fell off page 1 in the current window (avg rank > 10).

    Used by the ranking gate in `app/seo_diagnostic.py` as the "hybrid"
    threshold — we fire the gate on either the weighted-rank slip OR
    page-1 churn ≥ `PAGE1_CHURN_THRESHOLD`.

    Queries with very low prior impressions (< `min_prior_impressions`) are
    excluded so we don't count noisy long-tail terms. Each query is scoped
    by (domain, query) so the same text ranking for two sites is counted
    separately.

    Returns
    -------
    dict with `churn_pct` (0–1), `prior_page1_queries` (count),
    `churned_queries` (count), `examples` (up to 10 biggest click losers
    among the churned set).
    """
    dom_list = _gsc_domain_list(domains)
    query = f"""
    WITH prior AS (
      SELECT
        domain,
        LOWER(TRIM(query)) AS query,
        SUM(clicks)       AS clicks,
        SUM(impressions)  AS impressions,
        SUM(position * impressions) / NULLIF(SUM(impressions), 0) AS avg_rank
      FROM lakehouse_production.common.gsc_search_analytics_d_1
      WHERE date BETWEEN '{prior_start}' AND '{prior_end}'
        AND domain IN ({dom_list})
      GROUP BY domain, LOWER(TRIM(query))
      HAVING SUM(impressions) >= {int(min_prior_impressions)}
    ),
    curr AS (
      SELECT
        domain,
        LOWER(TRIM(query)) AS query,
        SUM(clicks)       AS clicks,
        SUM(impressions)  AS impressions,
        SUM(position * impressions) / NULLIF(SUM(impressions), 0) AS avg_rank
      FROM lakehouse_production.common.gsc_search_analytics_d_1
      WHERE date BETWEEN '{curr_start}' AND '{curr_end}'
        AND domain IN ({dom_list})
      GROUP BY domain, LOWER(TRIM(query))
    )
    SELECT
      p.domain,
      p.query,
      p.avg_rank AS prior_rank,
      c.avg_rank AS curr_rank,
      p.clicks   AS prior_clicks,
      COALESCE(c.clicks, 0) AS curr_clicks
    FROM prior p
    LEFT JOIN curr c
      ON p.domain = c.domain AND p.query = c.query
    WHERE p.avg_rank <= 4
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return {
            "churn_pct": None, "prior_page1_queries": 0,
            "churned_queries": 0, "examples": [],
        }

    for c in ["prior_rank", "curr_rank", "prior_clicks", "curr_clicks"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # A query "churns" if its current avg rank is past page 1 (>10) or if
    # it doesn't rank at all anymore (curr_rank is NaN).
    churned_mask = df["curr_rank"].isna() | (df["curr_rank"] > 10)
    total = int(len(df))
    churned = int(churned_mask.sum())
    churn_pct = (churned / total) if total else None

    examples_df = (
        df[churned_mask]
        .assign(click_delta=lambda x: x["curr_clicks"] - x["prior_clicks"])
        .sort_values("click_delta")
        .head(10)
    )
    examples = [
        {
            "domain": str(r["domain"]),
            "query": str(r["query"]),
            "prior_rank": float(r["prior_rank"]) if pd.notna(r["prior_rank"]) else None,
            "curr_rank": float(r["curr_rank"]) if pd.notna(r["curr_rank"]) else None,
            "prior_clicks": int(r["prior_clicks"]),
            "curr_clicks": int(r["curr_clicks"]),
            "click_delta": int(r["click_delta"]),
        }
        for _, r in examples_df.iterrows()
    ]

    return {
        "churn_pct": churn_pct,
        "prior_page1_queries": total,
        "churned_queries": churned,
        "examples": examples,
    }


@st.cache_data(ttl=7200, show_spinner="Finding pages with the biggest click/impression drops…")
def fetch_gsc_top_declining_pages(
    curr_start: date,
    curr_end: date,
    prior_start: date,
    prior_end: date,
    domains: list[str] | None = None,
    top_n: int = 15,
    min_prior_clicks: int = 50,
) -> pd.DataFrame:
    """
    Cross-site table of pages (URLs) that lost the most clicks between the
    prior and current windows. Sourced from `gsc_search_analytics_d_1`
    (see module docstring for the d_1 == d_3 empirical equivalence at
    (date, domain, page) grain).

    Returned columns (one row per page, sorted by `click_delta` ascending):
      site, domain, page, clicks_curr, clicks_prior, click_delta,
      click_delta_pct, impressions_curr, impressions_prior, impression_delta,
      ctr_curr, ctr_prior

    Only pages with `prior_clicks >= min_prior_clicks` are considered — this
    filters out noise from tiny URLs. Pages that existed only in the prior
    window (disappeared in the current window) are still included: their
    `clicks_curr` is 0.
    """
    dom_list = _gsc_domain_list(domains)
    query = f"""
    WITH curr AS (
      SELECT
        domain,
        page,
        SUM(clicks)      AS clicks,
        SUM(impressions) AS impressions
      FROM lakehouse_production.common.gsc_search_analytics_d_1
      WHERE date BETWEEN '{curr_start}' AND '{curr_end}'
        AND domain IN ({dom_list})
      GROUP BY domain, page
    ),
    prior AS (
      SELECT
        domain,
        page,
        SUM(clicks)      AS clicks,
        SUM(impressions) AS impressions
      FROM lakehouse_production.common.gsc_search_analytics_d_1
      WHERE date BETWEEN '{prior_start}' AND '{prior_end}'
        AND domain IN ({dom_list})
      GROUP BY domain, page
    )
    SELECT
      p.domain,
      p.page,
      COALESCE(c.clicks, 0)       AS clicks_curr,
      p.clicks                    AS clicks_prior,
      COALESCE(c.clicks, 0) - p.clicks AS click_delta,
      COALESCE(c.impressions, 0)  AS impressions_curr,
      p.impressions               AS impressions_prior,
      COALESCE(c.impressions, 0) - p.impressions AS impression_delta,
      CASE WHEN COALESCE(c.impressions, 0) > 0
           THEN COALESCE(c.clicks, 0) * 1.0 / c.impressions END AS ctr_curr,
      CASE WHEN p.impressions > 0
           THEN p.clicks * 1.0 / p.impressions END AS ctr_prior
    FROM prior p
    LEFT JOIN curr c
      ON p.domain = c.domain AND p.page = c.page
    WHERE p.clicks >= {int(min_prior_clicks)}
    ORDER BY click_delta ASC
    LIMIT {int(top_n)}
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["site"] = df["domain"].map(GSC_DOMAIN_TO_SITE).fillna(df["domain"])
    for c in [
        "clicks_curr", "clicks_prior", "click_delta",
        "impressions_curr", "impressions_prior", "impression_delta",
        "ctr_curr", "ctr_prior",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Percent change vs prior — safe divide.
    df["click_delta_pct"] = df.apply(
        lambda r: (r["click_delta"] / r["clicks_prior"]) if r["clicks_prior"] else None,
        axis=1,
    )
    df["impression_delta_pct"] = df.apply(
        lambda r: (r["impression_delta"] / r["impressions_prior"])
        if r["impressions_prior"] else None,
        axis=1,
    )

    return df[[
        "site", "domain", "page",
        "clicks_curr", "clicks_prior", "click_delta", "click_delta_pct",
        "impressions_curr", "impressions_prior", "impression_delta",
        "impression_delta_pct",
        "ctr_curr", "ctr_prior",
    ]]


@st.cache_data(ttl=7200, show_spinner="Loading GSC site-level trends…")
def fetch_gsc_site_trends(
    start_date: str,
    end_date: str,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    """
    Daily clicks / impressions / CTR / weighted-avg-rank from
    `gsc_search_analytics_d_5` (the site-level table that matches the GSC
    dashboard exactly — no `page` or `query` columns).

    Returns one row per (date, domain, site) with numeric totals; CTR and
    position are recomputed as impression-weighted aggregates rather than
    trusted as stored (the stored `ctr` and `position` are per-row, not
    aggregatable).

    Parameters
    ----------
    start_date, end_date : 'YYYY-MM-DD' strings, inclusive.
    domains : optional list of GSC domains (bare, no `www.`). Defaults to the
        four Texas brands.
    """
    dom_list = _gsc_domain_list(domains)
    query = f"""
    SELECT
      date,
      domain,
      SUM(clicks)                                           AS clicks,
      SUM(impressions)                                      AS impressions,
      SUM(clicks) * 1.0 / NULLIF(SUM(impressions), 0)       AS ctr,
      SUM(position * impressions) / NULLIF(SUM(impressions), 0) AS weighted_avg_rank
    FROM lakehouse_production.common.gsc_search_analytics_d_5
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
      AND domain IN ({dom_list})
    GROUP BY date, domain
    ORDER BY date, domain
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ["clicks", "impressions", "ctr", "weighted_avg_rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["site"] = df["domain"].map(GSC_DOMAIN_TO_SITE).fillna(df["domain"])
    return df


@st.cache_data(ttl=7200, show_spinner="Loading GSC page-type breakdown…")
def fetch_gsc_by_page_type(
    start_date: str,
    end_date: str,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    """
    Clicks / impressions / CTR / weighted-avg-rank grouped by
    `site × landing_page_type` for the supplied window, using
    `gsc_search_analytics_d_1` joined to the `url_to_page_type` CTE.

    Sourced from d_1 (see module docstring for the d_1 == d_3 empirical
    equivalence). Group-by keys stay at (domain, landing_page_type) — we
    do NOT add `query` to the GROUP BY; each page-type row should still
    represent all queries that landed on that page-type.

    GSC pages that don't appear in any organic session are bucketed as
    'Unmatched' — this is expected and useful (the page still earns
    impressions even without clicks).

    Returns columns:
      site, domain, landing_page_type, landing_page_type_bucket,
      clicks, impressions, ctr, weighted_avg_rank

    ``landing_page_type_bucket`` is the exec-view rollup (6 named buckets
    + Unmatched) defined by ``bucket_for_landing_page_type``. Downstream
    consumers can group on either column — the raw value for drill-down,
    the bucket for high-level views.
    """
    dom_list = _gsc_domain_list(domains)
    url_cte = _build_url_to_page_type_cte()

    raw_lpt_expr = "COALESCE(u.landing_page_type, 'Unmatched')"
    bucket_expr = _sql_bucket_case_expr(raw_lpt_expr, alias=None)

    query = f"""
    WITH {url_cte}
    SELECT
      CASE g.domain
        WHEN 'choosetexaspower.org'   THEN 'CTXP'
        WHEN 'saveonenergy.com'       THEN 'SOE'
        WHEN 'chooseenergy.com'       THEN 'Choose TX'
        WHEN 'texaselectricrates.com' THEN 'TXER'
        ELSE g.domain
      END                                                          AS site,
      g.domain,
      {raw_lpt_expr}                                               AS landing_page_type,
      {bucket_expr}                                                AS landing_page_type_bucket,
      SUM(g.clicks)                                                AS clicks,
      SUM(g.impressions)                                           AS impressions,
      SUM(g.clicks) * 1.0 / NULLIF(SUM(g.impressions), 0)          AS ctr,
      SUM(g.position * g.impressions) / NULLIF(SUM(g.impressions), 0) AS weighted_avg_rank
    FROM lakehouse_production.common.gsc_search_analytics_d_1 g
    LEFT JOIN url_to_page_type u
      ON RTRIM('/',
           LOWER(
             CASE WHEN POSITION('#' IN g.page) > 0
               THEN LEFT(g.page, POSITION('#' IN g.page) - 1)
               ELSE g.page
             END
           )
         ) = u.landing_page
    WHERE g.date BETWEEN '{start_date}' AND '{end_date}'
      AND g.domain IN ({dom_list})
    GROUP BY g.domain, {raw_lpt_expr}
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in ["clicks", "impressions", "ctr", "weighted_avg_rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=7200, show_spinner="Loading unmatched GSC URLs…")
def fetch_gsc_unmatched_urls(
    start_date: str,
    end_date: str,
    top_n: int = 20,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    """
    Top-N GSC URLs that fail to join to the `url_to_page_type` lookup — i.e.
    pages that have GSC impressions but don't appear as an Organic
    `first_page_url` in `mp_session_level_query` since 2025-01-01.

    Sourced from `gsc_search_analytics_d_1` (see module docstring for the
    d_1 == d_3 empirical equivalence at (date, domain, page) grain).

    Common causes (most impactful first):
      1. Page gets impressions but zero/few organic clicks (SERP-only visibility).
      2. Page is filtered OUT of the session table by the organic-session WHERE
         clause (non-Texas region, `/resources/`, `/solar-energy/`, company_id,
         bot filter, etc. — see `session_level_query`).
      3. URL normalisation drift between the two sources (trailing slash,
         uppercase, query strings, anchors — we already RTRIM('/'), LOWER(),
         and strip #fragments on both sides).
      4. The page was newly published after 2025-01-01 and hasn't accrued
         enough organic sessions to rank it yet.

    Returns `domain`, `site`, `page`, `clicks`, `impressions`, `ctr`,
    `weighted_avg_rank`, ordered by clicks desc.
    """
    dom_list = _gsc_domain_list(domains)
    url_cte = _build_url_to_page_type_cte()

    query = f"""
    WITH {url_cte},
    gsc_norm AS (
      SELECT
        g.domain,
        CASE g.domain
          WHEN 'choosetexaspower.org'   THEN 'CTXP'
          WHEN 'saveonenergy.com'       THEN 'SOE'
          WHEN 'chooseenergy.com'       THEN 'Choose TX'
          WHEN 'texaselectricrates.com' THEN 'TXER'
          ELSE g.domain
        END AS site,
        g.page AS raw_page,
        RTRIM('/',
          LOWER(
            CASE WHEN POSITION('#' IN g.page) > 0
              THEN LEFT(g.page, POSITION('#' IN g.page) - 1)
              ELSE g.page
            END
          )
        ) AS landing_page,
        g.clicks,
        g.impressions,
        g.position
      FROM lakehouse_production.common.gsc_search_analytics_d_1 g
      WHERE g.date BETWEEN '{start_date}' AND '{end_date}'
        AND g.domain IN ({dom_list})
    )
    SELECT
      gn.domain,
      gn.site,
      gn.raw_page                                                      AS page,
      SUM(gn.clicks)                                                   AS clicks,
      SUM(gn.impressions)                                              AS impressions,
      SUM(gn.clicks) * 1.0 / NULLIF(SUM(gn.impressions), 0)            AS ctr,
      SUM(gn.position * gn.impressions) / NULLIF(SUM(gn.impressions), 0) AS weighted_avg_rank
    FROM gsc_norm gn
    LEFT JOIN url_to_page_type u ON gn.landing_page = u.landing_page
    WHERE u.landing_page IS NULL
    GROUP BY gn.domain, gn.site, gn.raw_page
    ORDER BY clicks DESC
    LIMIT {int(top_n)}
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in ["clicks", "impressions", "ctr", "weighted_avg_rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # By definition these URLs failed the page-type lookup → they all live in
    # the hygiene bucket. Including the column keeps every GSC fetcher
    # symmetric for callers.
    df["landing_page_type"] = UNMATCHED_LANDING_PAGE_TYPE
    df["landing_page_type_bucket"] = UNMATCHED_BUCKET
    return df


@st.cache_data(ttl=7200, show_spinner="Loading top GSC queries by page type…")
def fetch_gsc_top_queries_by_page_type(
    start_date: str,
    end_date: str,
    top_n: int = 10,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    """
    Top-N GSC queries (by clicks) for each `site × landing_page_type` cell,
    sourced from `gsc_search_analytics_d_1` joined to `url_to_page_type`.

    `d_1` totals will NOT match the GSC dashboard because Google drops
    anonymised long-tail queries — see the PRD's GSC Table Selection Guide.
    Use this only to answer "what queries drive this page type?".
    """
    dom_list = _gsc_domain_list(domains)
    url_cte = _build_url_to_page_type_cte()

    query = f"""
    WITH {url_cte},
    gsc_norm AS (
      SELECT
        CASE g.domain
          WHEN 'choosetexaspower.org'   THEN 'CTXP'
          WHEN 'saveonenergy.com'       THEN 'SOE'
          WHEN 'chooseenergy.com'       THEN 'Choose TX'
          WHEN 'texaselectricrates.com' THEN 'TXER'
          ELSE g.domain
        END                                                        AS site,
        RTRIM('/',
          LOWER(
            CASE WHEN POSITION('#' IN g.page) > 0
              THEN LEFT(g.page, POSITION('#' IN g.page) - 1)
              ELSE g.page
            END
          )
        )                                                          AS landing_page,
        LOWER(TRIM(g.query))                                       AS query,
        g.clicks,
        g.impressions,
        g.position
      FROM lakehouse_production.common.gsc_search_analytics_d_1 g
      WHERE g.date BETWEEN '{start_date}' AND '{end_date}'
        AND g.domain IN ({dom_list})
    ),
    tagged AS (
      SELECT
        gn.site,
        COALESCE(u.landing_page_type, 'Unmatched') AS landing_page_type,
        gn.query,
        SUM(gn.clicks)       AS clicks,
        SUM(gn.impressions)  AS impressions,
        SUM(gn.position * gn.impressions) / NULLIF(SUM(gn.impressions), 0) AS avg_rank
      FROM gsc_norm gn
      LEFT JOIN url_to_page_type u ON gn.landing_page = u.landing_page
      GROUP BY gn.site, COALESCE(u.landing_page_type, 'Unmatched'), gn.query
    ),
    ranked AS (
      SELECT *,
        ROW_NUMBER() OVER (
          PARTITION BY site, landing_page_type ORDER BY clicks DESC
        ) AS rn
      FROM tagged
    )
    SELECT site, landing_page_type, query, clicks, impressions, avg_rank
    FROM ranked
    WHERE rn <= {int(top_n)}
    ORDER BY site, landing_page_type, clicks DESC
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in ["clicks", "impressions", "avg_rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["landing_page_type_bucket"] = df["landing_page_type"].map(
        bucket_for_landing_page_type
    )
    return df


@st.cache_data(ttl=7200, show_spinner="Tracking top keyword rankings by page type…")
def fetch_gsc_top_keyword_tracker(
    curr_start: str,
    curr_end: str,
    prior_start: str,
    prior_end: str,
    top_n: int = 5,
    min_prior_impressions: int = 500,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    """
    Rank-tracker signal for the TL;DR and the new "Top Keyword Ranking
    Tracker" section.

    For each ``landing_page_type``, picks the ``top_n`` queries by
    **prior-window clicks** (i.e. the keywords that mattered going into the
    current window) and returns both windows' clicks, impressions and
    impression-weighted avg rank side-by-side. A prior-impressions floor
    (``min_prior_impressions``) filters out long-tail noise — we only want
    queries that were materially trafficked in the prior window so a rank
    move actually means something.

    Page-type is attributed to each query via the ``landing_page`` that
    received the most prior-window clicks for that query (per domain). If
    the same keyword lands on multiple URLs, we tie-break to the URL that
    drove the most clicks across the full range. That avoids a keyword
    double-counting across page-types. Queries that map only to
    ``Unmatched`` URLs are dropped so the section focuses on templates we
    can actually take action on.

    Returns one row per (site, landing_page_type, query) with:
        site, domain, landing_page_type, query,
        prior_clicks, prior_impressions, prior_rank,
        curr_clicks,  curr_impressions,  curr_rank,
        click_delta, click_delta_pct, rank_delta,
        prior_rank_rank (1..top_n, for display ordering)

    ``rank_delta`` is ``curr_rank - prior_rank`` (positive = worse) and is
    ``NaN`` when the keyword has no current impressions (effectively
    dropped out). The render layer treats NaN as the worst possible mover
    and surfaces it explicitly.
    """
    dom_list = _gsc_domain_list(domains)
    url_cte = _build_url_to_page_type_cte()

    query = f"""
    WITH {url_cte},
    gsc_norm AS (
      SELECT
        CASE g.domain
          WHEN 'choosetexaspower.org'   THEN 'CTXP'
          WHEN 'saveonenergy.com'       THEN 'SOE'
          WHEN 'chooseenergy.com'       THEN 'Choose TX'
          WHEN 'texaselectricrates.com' THEN 'TXER'
          ELSE g.domain
        END                                                            AS site,
        g.domain,
        RTRIM('/',
          LOWER(
            CASE WHEN POSITION('#' IN g.page) > 0
              THEN LEFT(g.page, POSITION('#' IN g.page) - 1)
              ELSE g.page
            END
          )
        )                                                              AS landing_page,
        LOWER(TRIM(g.query))                                           AS query,
        g.date,
        g.clicks,
        g.impressions,
        g.position
      FROM lakehouse_production.common.gsc_search_analytics_d_1 g
      WHERE g.date BETWEEN '{prior_start}' AND '{curr_end}'
        AND g.domain IN ({dom_list})
    ),
    tagged AS (
      -- Attribute each (domain, query) to the landing_page_type that
      -- received the most PRIOR-window clicks for that query. Ties broken
      -- by total clicks across the full range so we pick a stable URL.
      SELECT
        gn.site,
        gn.domain,
        gn.query,
        COALESCE(u.landing_page_type, 'Unmatched') AS landing_page_type,
        SUM(CASE WHEN gn.date BETWEEN '{prior_start}' AND '{prior_end}'
                 THEN gn.clicks ELSE 0 END) AS prior_clicks_on_url,
        SUM(gn.clicks) AS total_clicks_on_url
      FROM gsc_norm gn
      LEFT JOIN url_to_page_type u ON gn.landing_page = u.landing_page
      GROUP BY gn.site, gn.domain, gn.query,
               COALESCE(u.landing_page_type, 'Unmatched')
    ),
    query_to_type AS (
      SELECT site, domain, query, landing_page_type
      FROM (
        SELECT
          site, domain, query, landing_page_type,
          ROW_NUMBER() OVER (
            PARTITION BY domain, query
            ORDER BY prior_clicks_on_url DESC, total_clicks_on_url DESC
          ) AS rn
        FROM tagged
      )
      WHERE rn = 1
    ),
    prior_agg AS (
      SELECT
        gn.domain,
        gn.query,
        SUM(gn.clicks)                                                 AS prior_clicks,
        SUM(gn.impressions)                                            AS prior_impressions,
        SUM(gn.position * gn.impressions) /
          NULLIF(SUM(gn.impressions), 0)                               AS prior_rank
      FROM gsc_norm gn
      WHERE gn.date BETWEEN '{prior_start}' AND '{prior_end}'
      GROUP BY gn.domain, gn.query
    ),
    curr_agg AS (
      SELECT
        gn.domain,
        gn.query,
        SUM(gn.clicks)                                                 AS curr_clicks,
        SUM(gn.impressions)                                            AS curr_impressions,
        SUM(gn.position * gn.impressions) /
          NULLIF(SUM(gn.impressions), 0)                               AS curr_rank
      FROM gsc_norm gn
      WHERE gn.date BETWEEN '{curr_start}' AND '{curr_end}'
      GROUP BY gn.domain, gn.query
    ),
    joined AS (
      SELECT
        qt.site,
        qt.domain,
        qt.landing_page_type,
        qt.query,
        COALESCE(p.prior_clicks, 0)       AS prior_clicks,
        COALESCE(p.prior_impressions, 0)  AS prior_impressions,
        p.prior_rank,
        COALESCE(c.curr_clicks, 0)        AS curr_clicks,
        COALESCE(c.curr_impressions, 0)   AS curr_impressions,
        c.curr_rank
      FROM query_to_type qt
      LEFT JOIN prior_agg p ON qt.domain = p.domain AND qt.query = p.query
      LEFT JOIN curr_agg  c ON qt.domain = c.domain AND qt.query = c.query
      WHERE qt.landing_page_type <> 'Unmatched'
        AND COALESCE(p.prior_impressions, 0) >= {int(min_prior_impressions)}
    ),
    ranked AS (
      SELECT
        j.*,
        ROW_NUMBER() OVER (
          PARTITION BY site, landing_page_type
          ORDER BY prior_clicks DESC, prior_impressions DESC
        ) AS prior_rank_rank
      FROM joined j
    )
    SELECT
      site,
      domain,
      landing_page_type,
      query,
      prior_clicks,
      prior_impressions,
      prior_rank,
      curr_clicks,
      curr_impressions,
      curr_rank,
      prior_rank_rank
    FROM ranked
    WHERE prior_rank_rank <= {int(top_n)}
    ORDER BY site, landing_page_type, prior_rank_rank
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in [
        "prior_clicks", "prior_impressions", "prior_rank",
        "curr_clicks", "curr_impressions", "curr_rank",
        "prior_rank_rank",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["click_delta"] = df["curr_clicks"] - df["prior_clicks"]
    df["click_delta_pct"] = df.apply(
        lambda r: (r["click_delta"] / r["prior_clicks"])
        if r["prior_clicks"] else None,
        axis=1,
    )
    df["rank_delta"] = df["curr_rank"] - df["prior_rank"]
    df["landing_page_type_bucket"] = df["landing_page_type"].map(
        bucket_for_landing_page_type
    )
    return df


@st.cache_data(ttl=7200, show_spinner="Loading Organic session funnel by page type…")
def fetch_organic_session_funnel_by_page_type(
    start_date: str,
    end_date: str,
    websites: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Organic-only session funnel from `mp_session_level_query`, grouped by
    `website (site) × landing_page_type`. Returns raw aggregates plus
    derived rates (zlur_pct, cart_rate_pct, vc_pct, phone_rr_pct,
    phone_vc_pct, cart_conversion_pct, cart_vc_pct).

    Pass ``websites`` as a tuple of `website` values (e.g. ``("CTXP",)``)
    to restrict to a single brand. ``None`` returns every Texas brand —
    callers can still filter in pandas after the fact.

    Used to pair Tier-3 GSC visibility (above) with the site's own funnel
    performance for a given page type. Single channel — Organic only — so
    the session-to-GSC reconciliation stays tight.
    """
    site_filter = ""
    if websites:
        literals = ", ".join(f"'{w}'" for w in websites)
        site_filter = f"AND website IN ({literals})"
    query = f"""
    SELECT
      website                                    AS site,
      landing_page_type,
      SUM(sessions)                              AS sessions,
      SUM(zip_entry)                             AS zip_entries,
      SUM(has_cart)                              AS carts,
      SUM(cart_orders) + SUM(phone_orders)       AS orders,
      SUM(cart_orders)                           AS cart_orders,
      SUM(phone_orders)                          AS phone_orders,
      SUM(queue_calls)                           AS queue_calls
    FROM energy_prod.data_science.mp_session_level_query
    WHERE marketing_channel = 'Organic'
      AND _date BETWEEN '{start_date}' AND '{end_date}'
      {site_filter}
    GROUP BY website, landing_page_type
    HAVING SUM(sessions) >= 10
    ORDER BY site, sessions DESC
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in [
        "sessions", "zip_entries", "carts", "orders",
        "cart_orders", "phone_orders", "queue_calls",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["zlur_pct"] = df["zip_entries"] / df["sessions"].replace(0, pd.NA)
    df["cart_rate_pct"] = df["carts"] / df["sessions"].replace(0, pd.NA)
    df["vc_pct"] = df["orders"] / df["sessions"].replace(0, pd.NA)
    # Phone-funnel rates from README: Phone RR = queue_calls / sessions,
    # Phone VC (aka Phone Conv) = phone_orders / sessions.
    df["phone_rr_pct"] = df["queue_calls"] / df["sessions"].replace(0, pd.NA)
    df["phone_vc_pct"] = df["phone_orders"] / df["sessions"].replace(0, pd.NA)
    # Cart Conversion = cart_orders / carts (orders per cart entry).
    df["cart_conversion_pct"] = df["cart_orders"] / df["carts"].replace(0, pd.NA)
    df["cart_vc_pct"] = df["cart_orders"] / df["sessions"].replace(0, pd.NA)
    df["landing_page_type_bucket"] = df["landing_page_type"].map(
        bucket_for_landing_page_type
    )
    return df


# ---------------------------------------------------------------------------
# Bucket aggregation helpers — used by the Organic Deep Dive tab + bot tools
# to roll raw `landing_page_type` rows up to the 6-bucket taxonomy without
# losing the underlying numerators. All impression-weighted metrics (CTR,
# rank) are recomputed from the raw numerators so the output stays
# aggregatable.
# ---------------------------------------------------------------------------


def rollup_gsc_to_bucket(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Sum GSC metrics (clicks / impressions) up to ``landing_page_type_bucket``
    and recompute CTR + impression-weighted rank from the raw totals.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``landing_page_type``, ``clicks``,
        ``impressions``, ``weighted_avg_rank``. If the bucket column is
        missing it's computed on the fly from ``landing_page_type``.
    group_cols : optional list of columns to keep as additional grouping
        keys (e.g. ``["site"]`` to preserve per-site breakdowns). The
        bucket column is always added to the group keys.

    Returns
    -------
    pd.DataFrame with ``landing_page_type_bucket`` + any extra group cols +
    ``clicks``, ``impressions``, ``ctr``, ``weighted_avg_rank``.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "landing_page_type_bucket", "clicks", "impressions",
            "ctr", "weighted_avg_rank",
        ])

    tmp = df.copy()
    if "landing_page_type_bucket" not in tmp.columns:
        tmp["landing_page_type_bucket"] = tmp["landing_page_type"].map(
            bucket_for_landing_page_type
        )
    # Guard against NaN rank on zero-impression cells before the multiply.
    tmp["pos_x_impr"] = (
        pd.to_numeric(tmp["weighted_avg_rank"], errors="coerce").fillna(0)
        * pd.to_numeric(tmp["impressions"], errors="coerce").fillna(0)
    )

    keys = ["landing_page_type_bucket"] + list(group_cols or [])
    agg = (
        tmp.groupby(keys, as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            pos_x_impr=("pos_x_impr", "sum"),
        )
    )
    agg["ctr"] = agg["clicks"] / agg["impressions"].replace(0, pd.NA)
    agg["weighted_avg_rank"] = (
        agg["pos_x_impr"] / agg["impressions"].replace(0, pd.NA)
    )
    return agg.drop(columns=["pos_x_impr"])


def rollup_sessions_to_bucket(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Sum session-funnel counts up to ``landing_page_type_bucket`` and
    recompute rate columns (zlur_pct / cart_rate_pct / vc_pct) from the
    raw totals.

    Input: output of ``fetch_organic_session_funnel_by_page_type`` (or any
    frame with ``sessions`` / ``orders`` / optionally ``zip_entries`` /
    ``carts`` / ``cart_orders`` / ``phone_orders``).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "landing_page_type_bucket", "sessions", "orders",
            "zlur_pct", "cart_rate_pct", "vc_pct",
        ])

    tmp = df.copy()
    if "landing_page_type_bucket" not in tmp.columns:
        tmp["landing_page_type_bucket"] = tmp["landing_page_type"].map(
            bucket_for_landing_page_type
        )

    keys = ["landing_page_type_bucket"] + list(group_cols or [])
    sum_cols = [
        c for c in [
            "sessions", "zip_entries", "carts", "orders",
            "cart_orders", "phone_orders", "queue_calls",
        ] if c in tmp.columns
    ]
    agg = tmp.groupby(keys, as_index=False)[sum_cols].sum()

    if "sessions" in agg.columns:
        safe_sessions = agg["sessions"].replace(0, pd.NA)
        if "zip_entries" in agg.columns:
            agg["zlur_pct"] = agg["zip_entries"] / safe_sessions
        if "carts" in agg.columns:
            agg["cart_rate_pct"] = agg["carts"] / safe_sessions
        if "orders" in agg.columns:
            agg["vc_pct"] = agg["orders"] / safe_sessions
        if "queue_calls" in agg.columns:
            agg["phone_rr_pct"] = agg["queue_calls"] / safe_sessions
        if "phone_orders" in agg.columns:
            agg["phone_vc_pct"] = agg["phone_orders"] / safe_sessions
        if "cart_orders" in agg.columns:
            agg["cart_vc_pct"] = agg["cart_orders"] / safe_sessions
    if "cart_orders" in agg.columns and "carts" in agg.columns:
        agg["cart_conversion_pct"] = agg["cart_orders"] / agg["carts"].replace(0, pd.NA)
    return agg


# ---------------------------------------------------------------------------
# Pandas helpers used by the Organic Deep Dive tab (aggregations only —
# nothing touching the network).
# ---------------------------------------------------------------------------


def aggregate_gsc_daily_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Roll daily GSC site-level trends into monthly totals with paced
    partial-month projections.

    Input columns: date, site, clicks, impressions (as returned by
    `fetch_gsc_site_trends`). Output adds `month`, `days_with_data`,
    `days_in_month`, `clicks_paced`, `impressions_paced`, `ctr`, and
    `weighted_avg_rank` (recomputed from the aggregated totals).
    """
    if df_daily.empty:
        return df_daily

    tmp = df_daily.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["month"] = tmp["date"].dt.to_period("M").dt.start_time.dt.date
    tmp["days_in_month"] = tmp["date"].dt.daysinmonth

    tmp["pos_x_impr"] = tmp["weighted_avg_rank"] * tmp["impressions"]

    agg = (
        tmp.groupby(["month", "site"])
        .agg(
            days_with_data=("date", "nunique"),
            days_in_month=("days_in_month", "max"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            pos_x_impr=("pos_x_impr", "sum"),
        )
        .reset_index()
    )

    agg["ctr"] = agg["clicks"] / agg["impressions"].replace(0, pd.NA)
    agg["weighted_avg_rank"] = agg["pos_x_impr"] / agg["impressions"].replace(0, pd.NA)

    partial = agg["days_with_data"] < agg["days_in_month"]
    pace_factor = agg["days_in_month"] / agg["days_with_data"].replace(0, pd.NA)
    agg["clicks_paced"] = agg["clicks"].where(~partial, agg["clicks"] * pace_factor)
    agg["impressions_paced"] = agg["impressions"].where(
        ~partial, agg["impressions"] * pace_factor
    )
    agg["is_partial"] = partial

    return agg.drop(columns=["pos_x_impr"]).sort_values(["month", "site"])


def compute_click_decomposition(
    curr_clicks: float,
    curr_impressions: float,
    prior_clicks: float,
    prior_impressions: float,
) -> dict:
    """
    Additive decomposition of click deltas into impression-effect and CTR-effect.

    Uses the textbook formulation from `.cursor/rules/seo-reporting-agent.mdc`:
        impression_effect = (Δ impressions) × prior_ctr
        ctr_effect        = (Δ CTR)         × prior_impressions
        interaction       = (Δ impressions) × (Δ CTR)   [residual; small]

    Returns absolute clicks, CTR, and the three decomposition components in a
    single dict so the tab can render the narrative + diagnostic tree.
    """
    prior_ctr = (prior_clicks / prior_impressions) if prior_impressions else 0.0
    curr_ctr = (curr_clicks / curr_impressions) if curr_impressions else 0.0
    delta_impr = curr_impressions - prior_impressions
    delta_ctr = curr_ctr - prior_ctr
    delta_clicks = curr_clicks - prior_clicks

    impression_effect = delta_impr * prior_ctr
    ctr_effect = delta_ctr * prior_impressions
    interaction = delta_impr * delta_ctr

    return {
        "prior_clicks": prior_clicks,
        "curr_clicks": curr_clicks,
        "delta_clicks": delta_clicks,
        "prior_impressions": prior_impressions,
        "curr_impressions": curr_impressions,
        "delta_impressions": delta_impr,
        "prior_ctr": prior_ctr,
        "curr_ctr": curr_ctr,
        "delta_ctr": delta_ctr,
        "impression_effect": impression_effect,
        "ctr_effect": ctr_effect,
        "interaction": interaction,
        "pct_change_clicks": (delta_clicks / prior_clicks) if prior_clicks else 0.0,
        "pct_change_impressions": (delta_impr / prior_impressions) if prior_impressions else 0.0,
        "pct_change_ctr": (delta_ctr / prior_ctr) if prior_ctr else 0.0,
    }


def diagnose_click_change(
    decomp: dict,
    curr_rank: float | None = None,
    prior_rank: float | None = None,
    rank_threshold: float = 0.5,
) -> str:
    """
    Apply the diagnostic decision tree from the SEO reporting agent rule and
    return a short human-readable explanation.

    Priority order:
      1. Rankings changed materially (>= rank_threshold positions)?
      2. Impressions moved but rankings didn't → demand shift
      3. CTR moved but impressions didn't → SERP-feature compression
      4. Otherwise: mix shift / measurement — investigate further
    """
    rank_delta = None
    if curr_rank is not None and prior_rank is not None:
        rank_delta = curr_rank - prior_rank

    pct_impr = decomp["pct_change_impressions"]
    pct_ctr = decomp["pct_change_ctr"]
    impr_effect = decomp["impression_effect"]
    ctr_effect = decomp["ctr_effect"]

    # 1. Ranking moved
    if rank_delta is not None and abs(rank_delta) >= rank_threshold:
        direction = "regression" if rank_delta > 0 else "improvement"
        sign = "dropped" if rank_delta > 0 else "rose"
        return (
            f"**Ranking {direction} drove the click change.** "
            f"Weighted average rank {sign} by {abs(rank_delta):.1f} positions "
            f"(prior {prior_rank:.1f} → current {curr_rank:.1f}). "
            "Check for algorithm updates or competitor SERP changes."
        )

    # 2. Impressions moved, rankings stable
    if abs(impr_effect) > abs(ctr_effect) and abs(pct_impr) >= 0.03:
        direction = "declined" if pct_impr < 0 else "grew"
        return (
            f"**Search demand {direction}.** Rankings were stable but impressions "
            f"{direction} {abs(pct_impr) * 100:.1f}%, contributing roughly "
            f"{impr_effect:+,.0f} clicks. Likely seasonal or market-level shift."
        )

    # 3. CTR moved, impressions stable
    if abs(ctr_effect) > abs(impr_effect) and abs(pct_ctr) >= 0.03:
        direction = "fell" if pct_ctr < 0 else "rose"
        extra = (
            "Check for AI Overviews, featured snippets, or other SERP features "
            "compressing organic clicks."
            if pct_ctr < 0
            else "Likely SERP layout became friendlier — monitor sustainability."
        )
        return (
            f"**CTR {direction}.** Impressions held roughly steady but CTR {direction} "
            f"{abs(pct_ctr) * 100:.1f}%, contributing roughly "
            f"{ctr_effect:+,.0f} clicks. {extra}"
        )

    # 4. Default
    return (
        "**No dominant driver identified.** Impression and CTR effects are both "
        "small — this period's click movement looks like mix shift or noise. "
        "Drill into the page-type breakdown below to confirm."
    )
