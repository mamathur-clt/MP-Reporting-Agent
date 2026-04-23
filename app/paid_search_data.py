"""
Data layer for the Paid Search deep-dive tab.

Wraps `docs/Paid Query.sql` — the campaign-bucket rollup over
`energy_prod.energy.paidsearch_campaign` joined to sessions, carts, calls
and orders by `campaign_id`. That query returns one row per
`(day, campaign_bucket)` with raw marketing and funnel metrics; this
module reads the file, rewrites the query window + `:grain` binding to
match the caller's date range, executes it, and exposes aggregation
helpers for the Streamlit tab.

Three families of helpers live here:

1. **Fetchers** — `fetch_paid_daily` pulls the daily×bucket frame for a
   single date window. `fetch_paid_for_windows` runs it once with a
   window that spans both the current and prior period so the tab only
   pays one round-trip to Databricks per render.
2. **Aggregators** — `aggregate_bucket_period` sums the daily frame to a
   per-bucket period summary with the marketing-funnel rates derived
   correctly (click-weighted CTR, impression-weighted CPC, etc.).
3. **Comparators / decomposers** — `compare_bucket_periods` joins the
   current + prior period frames and attaches delta columns;
   `bucket_vc_decomposition` runs a counterfactual mix/performance
   decomposition of Visit Conversion by campaign bucket — session share
   shifts (mix) vs within-bucket VC changes (perf) — returning
   per-bucket pp contributions that sum exactly to the portfolio VC delta.

Campaign bucket mapping (`CAMPAIGN_BUCKETS`) is kept here as the single
source of truth — the SQL already classifies buckets, but the list is
replicated so the UI can order them consistently and so the bucket
registry is available without a DB round-trip.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Literal

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

_PAID_QUERY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs", "Paid Query.sql"
)

# ---------------------------------------------------------------------------
# Bucket registry
# ---------------------------------------------------------------------------
#
# Kept in the order we want buckets to render in the UI (largest / most
# strategically interesting first, fallthrough last). This ordering is
# *display* only — the SQL classifies each campaign independently.

CAMPAIGN_BUCKETS: list[str] = [
    "Brand",
    "Supplier",
    "Aggregator",
    "Companies",
    "Generic",
    "Geo",
    "Price Sensitive",
    "Spanish",
    "Rates",
    "NoDeposit",
    "PMax",
    "Other",
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _get_connection():
    return databricks_sql.connect(
        server_hostname=_HOST.replace("https://", "").strip("/"),
        http_path=_HTTP_PATH,
        access_token=_TOKEN,
    )


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------


Grain = Literal["day", "week", "month"]


def _load_paid_query_template() -> str:
    """Read `docs/Paid Query.sql` once per process — it's a static asset."""
    with open(_PAID_QUERY_PATH) as f:
        return f.read()


def _render_paid_query(start_date: date, end_date: date, grain: Grain) -> str:
    """
    Parameterize the Paid Query template.

    The upstream file hardcodes a 2025-01-01 → ``date_add(current_date, -1)``
    window inside the ``day_dates`` CTE and uses ``:grain`` as a bind
    variable for ``DATE_TRUNC``. databricks-sql-connector doesn't support
    binding a raw identifier (``day`` / ``week`` / ``month`` goes inside
    ``DATE_TRUNC(...)``), so we materialise both parameters as string
    substitutions up front.

    The caller owns the date range — we pass it through verbatim. The
    `:grain` bind is also replaced with a literal so Databricks sees a
    fully-formed query.
    """
    if grain not in ("day", "week", "month"):
        raise ValueError(f"grain must be one of 'day', 'week', 'month' — got {grain!r}")

    tmpl = _load_paid_query_template()

    # Replace the hardcoded sequence bounds in the `day_dates` CTE.
    old_seq = (
        "sequence(\n"
        "                to_date('2025-01-01'), \n"
        "                date_add(current_date, -1),\n"
        "                interval 1 day\n"
        "            )"
    )
    new_seq = (
        f"sequence(\n"
        f"                to_date('{start_date}'), \n"
        f"                to_date('{end_date}'),\n"
        f"                interval 1 day\n"
        f"            )"
    )
    if old_seq not in tmpl:
        raise RuntimeError(
            "Paid Query template has drifted — can't find the `day_dates` sequence "
            "block to rewrite. Update `_render_paid_query` to match the new shape."
        )
    tmpl = tmpl.replace(old_seq, new_seq)

    # Replace the `:grain` bind with a literal. DATE_TRUNC expects a
    # quoted string on Databricks.
    tmpl = tmpl.replace(":grain", f"'{grain}'")
    return tmpl


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


_NUMERIC_COLUMNS = [
    "impressioncount",
    "clickcount",
    "cost",
    "sessions",
    "cart_starts",
    "gross_calls",
    "queue_calls",
    "net_calls",
    "gross_serp",
    "queue_serp",
    "net_serp",
    "scc",
    "phone_orders",
    "cart_orders2",
    "serp_orders",
    "site_phone_orders",
    "est_rev",
    "site_queue_calls",
    "queue_calls_grid",
    "queue_calls_homepage",
    "queue_calls_other",
]


@st.cache_data(ttl=1800, show_spinner="Loading Paid Search campaign data…")
def fetch_paid_daily(
    start_date: str,
    end_date: str,
    grain: Grain = "day",
) -> pd.DataFrame:
    """
    Execute `docs/Paid Query.sql` for the given window and return a daily
    campaign-bucket DataFrame.

    Parameters
    ----------
    start_date, end_date : str (YYYY-MM-DD)
        Inclusive bounds for the `day_dates` CTE.
    grain : {"day", "week", "month"}
        Value substituted into the `DATE_TRUNC` expression that produces
        the ``date_grain`` column. The row grain is always daily; the
        ``date_grain`` column is a bucketed label the caller can use for
        roll-ups.

    Returns
    -------
    DataFrame
        Columns: ``day`` (date), ``campaign_bucket`` (str),
        ``date_grain`` (str), ``day_of_week`` (str), ``day_of_month``
        (int), plus every metric in `_NUMERIC_COLUMNS` as float.
    """
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    query = _render_paid_query(start, end, grain)

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["day"] = pd.to_datetime(df["day"]).dt.date
    for c in _NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    if "day_of_month" in df.columns:
        df["day_of_month"] = pd.to_numeric(df["day_of_month"], errors="coerce").fillna(0).astype(int)
    return df


def fetch_paid_for_windows(
    curr_start: date,
    curr_end: date,
    prior_start: date,
    prior_end: date,
) -> pd.DataFrame:
    """
    Pull a single daily×bucket frame spanning both the current and prior
    windows so the tab only issues one Databricks round-trip per render.

    The two windows may not be contiguous (e.g. MoM MTD → current week
    and last month's same-length window); this fetcher unions the two
    ranges into ``[min(curr_start, prior_start), max(curr_end, prior_end)]``
    which always covers both. Downstream helpers filter by date range.
    """
    start = min(curr_start, prior_start)
    end = max(curr_end, prior_end)
    return fetch_paid_daily(str(start), str(end), grain="day")


# ---------------------------------------------------------------------------
# Aggregation / derived metrics
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return num / den


def _derived_rates(agg: dict) -> dict:
    """
    Attach derived rates to an aggregated row dict.

    All rates are re-derived from the raw numerators/denominators — taking
    a mean of daily rates would over-weight low-traffic days. The dict is
    mutated in place AND returned for ergonomic chaining.
    """
    impressions = agg.get("impressioncount", 0.0) or 0.0
    clicks = agg.get("clickcount", 0.0) or 0.0
    cost = agg.get("cost", 0.0) or 0.0
    sessions = agg.get("sessions", 0.0) or 0.0
    cart_orders = agg.get("cart_orders2", 0.0) or 0.0
    phone_orders = agg.get("phone_orders", 0.0) or 0.0
    revenue = agg.get("est_rev", 0.0) or 0.0

    total_orders = cart_orders + phone_orders

    agg["ctr"] = _safe_div(clicks, impressions)
    agg["cpc"] = _safe_div(cost, clicks)
    agg["click_to_session"] = _safe_div(sessions, clicks)
    agg["cart_rr"] = _safe_div(agg.get("cart_starts", 0.0) or 0.0, sessions)
    agg["cart_conversion"] = _safe_div(cart_orders, agg.get("cart_starts", 0.0) or 0.0)
    agg["cart_vc"] = _safe_div(cart_orders, sessions)
    agg["phone_vc"] = _safe_div(phone_orders, sessions)
    agg["vc"] = _safe_div(total_orders, sessions)
    agg["total_orders"] = total_orders
    agg["cost_per_order"] = _safe_div(cost, total_orders)
    agg["cost_per_session"] = _safe_div(cost, sessions)
    agg["revenue_per_order"] = _safe_div(revenue, total_orders)
    agg["revenue_per_session"] = _safe_div(revenue, sessions)
    agg["roas"] = _safe_div(revenue, cost)
    return agg


def _slice_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Return rows where `day` is in [start, end] (inclusive)."""
    if df is None or df.empty:
        return df
    mask = (df["day"] >= start) & (df["day"] <= end)
    return df.loc[mask].copy()


def aggregate_bucket_period(
    daily_df: pd.DataFrame,
    start: date,
    end: date,
    *,
    buckets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Collapse a daily×bucket frame to one row per ``campaign_bucket`` over
    ``[start, end]`` with raw totals + derived rates.

    The row grain is one campaign bucket. Columns returned:
      - raw metrics from `_NUMERIC_COLUMNS`
      - derived rates from `_derived_rates` (ctr, cpc, cart_rr, vc, ROAS, …)
      - ``total_orders`` (cart + phone)
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["campaign_bucket", *_NUMERIC_COLUMNS])

    sub = _slice_window(daily_df, start, end)
    if buckets is not None:
        wanted = list(buckets)
        sub = sub[sub["campaign_bucket"].isin(wanted)]
    if sub.empty:
        return pd.DataFrame(columns=["campaign_bucket", *_NUMERIC_COLUMNS])

    cols = [c for c in _NUMERIC_COLUMNS if c in sub.columns]
    agg = sub.groupby("campaign_bucket", as_index=False)[cols].sum()

    rate_rows: list[dict] = []
    for _, row in agg.iterrows():
        rate_rows.append(_derived_rates(row.to_dict()))
    return pd.DataFrame(rate_rows)


def aggregate_total_period(
    daily_df: pd.DataFrame,
    start: date,
    end: date,
    *,
    buckets: Iterable[str] | None = None,
) -> dict:
    """All-bucket totals for a single period, as a plain dict."""
    if daily_df is None or daily_df.empty:
        return _derived_rates({c: 0.0 for c in _NUMERIC_COLUMNS})

    sub = _slice_window(daily_df, start, end)
    if buckets is not None:
        wanted = list(buckets)
        sub = sub[sub["campaign_bucket"].isin(wanted)]
    if sub.empty:
        return _derived_rates({c: 0.0 for c in _NUMERIC_COLUMNS})

    cols = [c for c in _NUMERIC_COLUMNS if c in sub.columns]
    totals = sub[cols].sum().to_dict()
    return _derived_rates(totals)


# ---------------------------------------------------------------------------
# Period-over-period comparison
# ---------------------------------------------------------------------------


def _pct(curr: float | None, prior: float | None) -> float | None:
    """Fractional change; None when prior is zero/missing."""
    if curr is None or prior is None or pd.isna(curr) or pd.isna(prior) or prior == 0:
        return None
    return (curr - prior) / prior


def compare_bucket_periods(
    daily_df: pd.DataFrame,
    curr_start: date,
    curr_end: date,
    prior_start: date,
    prior_end: date,
    *,
    buckets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Build the bucket-level comparison frame powering Section 3.

    One row per bucket (any bucket appearing in either period), with
    current / prior values and the fractional deltas for each metric the
    UI displays. Buckets are ordered per `CAMPAIGN_BUCKETS` (display
    order); any buckets present in the data but not in the registry are
    appended alphabetically at the end.
    """
    curr = aggregate_bucket_period(daily_df, curr_start, curr_end, buckets=buckets)
    prior = aggregate_bucket_period(daily_df, prior_start, prior_end, buckets=buckets)

    if curr.empty and prior.empty:
        return pd.DataFrame()

    metric_cols = [
        "impressioncount", "clickcount", "cost", "sessions", "cart_starts",
        "total_orders", "cart_orders2", "phone_orders", "est_rev",
        "ctr", "cpc", "cart_rr", "cart_conversion", "vc", "cost_per_order",
        "revenue_per_order", "roas",
    ]

    c = curr.set_index("campaign_bucket") if not curr.empty else curr
    p = prior.set_index("campaign_bucket") if not prior.empty else prior

    buckets_present = sorted(set(c.index).union(set(p.index)))
    ordered = [b for b in CAMPAIGN_BUCKETS if b in buckets_present] + [
        b for b in buckets_present if b not in CAMPAIGN_BUCKETS
    ]

    rows: list[dict] = []
    for bucket in ordered:
        row: dict = {"campaign_bucket": bucket}
        curr_row = c.loc[bucket] if (not c.empty and bucket in c.index) else None
        prior_row = p.loc[bucket] if (not p.empty and bucket in p.index) else None
        for m in metric_cols:
            curr_val = float(curr_row[m]) if curr_row is not None and m in curr_row else 0.0
            prior_val = float(prior_row[m]) if prior_row is not None and m in prior_row else 0.0
            row[f"{m}_curr"] = curr_val
            row[f"{m}_prior"] = prior_val
            row[f"{m}_delta"] = curr_val - prior_val
            row[f"{m}_delta_pct"] = _pct(curr_val, prior_val)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Campaign-bucket VC decomposition waterfall (Section 4)
# ---------------------------------------------------------------------------
#
# Counterfactual decomposition of the portfolio-level VC (visit
# conversion = orders / sessions) change into per-bucket mix and
# performance effects.
#
# Portfolio VC = Σ(w_b × vc_b) where w_b = bucket's session share.
# Period-over-period:
#   mix_impact_b  = (curr_w_b - prior_w_b) × prior_vc_b
#   perf_impact_b = curr_w_b × (curr_vc_b - prior_vc_b)
#   Σ(mix + perf) = curr_vc_portfolio - prior_vc_portfolio  (exact)
#
# This is the same framework used in the analyst's standalone SQL
# waterfall query.  Values are in VC (rate) units and displayed as
# percentage-point (pp) contributions.


@dataclass
class BucketVCDecompRow:
    """One bucket's contribution to the portfolio VC delta."""

    campaign_bucket: str
    curr_sessions: float
    prior_sessions: float
    curr_orders: float
    prior_orders: float
    curr_vc: float
    prior_vc: float
    curr_share: float
    prior_share: float

    mix_impact: float      # (Δw) × prior_vc  — in VC-rate units
    perf_impact: float     # curr_w × (Δvc)   — in VC-rate units

    @property
    def total_impact(self) -> float:
        return self.mix_impact + self.perf_impact

    @property
    def mix_impact_pp(self) -> float:
        return self.mix_impact * 100.0

    @property
    def perf_impact_pp(self) -> float:
        return self.perf_impact * 100.0

    @property
    def total_impact_pp(self) -> float:
        return self.total_impact * 100.0


@dataclass
class BucketVCDecompResult:
    """Packaged result of the bucket-level VC decomposition."""

    rows: list[BucketVCDecompRow]
    curr_vc_total: float
    prior_vc_total: float
    curr_sessions_total: float
    prior_sessions_total: float
    curr_orders_total: float
    prior_orders_total: float

    @property
    def vc_delta(self) -> float:
        return self.curr_vc_total - self.prior_vc_total

    @property
    def vc_delta_pp(self) -> float:
        return self.vc_delta * 100.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "campaign_bucket": r.campaign_bucket,
                "curr_sessions": r.curr_sessions,
                "prior_sessions": r.prior_sessions,
                "curr_orders": r.curr_orders,
                "prior_orders": r.prior_orders,
                "curr_vc": r.curr_vc,
                "prior_vc": r.prior_vc,
                "curr_share": r.curr_share,
                "prior_share": r.prior_share,
                "mix_impact": r.mix_impact,
                "perf_impact": r.perf_impact,
                "total_impact": r.total_impact,
                "mix_impact_pp": r.mix_impact_pp,
                "perf_impact_pp": r.perf_impact_pp,
                "total_impact_pp": r.total_impact_pp,
            }
            for r in self.rows
        ])


def bucket_vc_decomposition(
    daily_df: pd.DataFrame,
    curr_start: date,
    curr_end: date,
    prior_start: date,
    prior_end: date,
    *,
    buckets: Iterable[str] | None = None,
) -> BucketVCDecompResult | None:
    """
    Counterfactual mix / performance decomposition of the paid VC change
    sliced by ``campaign_bucket``.

    Portfolio VC = Σ(w_b × vc_b) where w_b is the bucket's session share
    and vc_b is the bucket's visit conversion (orders / sessions).

    For each bucket b the period-over-period contribution is:

        mix_impact_b  = (curr_w_b − prior_w_b) × prior_vc_b
        perf_impact_b = curr_w_b × (curr_vc_b − prior_vc_b)

    By construction Σ(mix + perf) = curr_vc − prior_vc exactly, so the
    waterfall reconciles to the total VC movement.

    Returns
    -------
    BucketVCDecompResult | None
        None when there's no data to decompose in either period.
    """
    curr_bucket = aggregate_bucket_period(daily_df, curr_start, curr_end, buckets=buckets)
    prior_bucket = aggregate_bucket_period(daily_df, prior_start, prior_end, buckets=buckets)

    if curr_bucket.empty and prior_bucket.empty:
        return None

    curr_tot_sessions = float(curr_bucket["sessions"].sum()) if not curr_bucket.empty else 0.0
    prior_tot_sessions = float(prior_bucket["sessions"].sum()) if not prior_bucket.empty else 0.0

    if curr_tot_sessions == 0 and prior_tot_sessions == 0:
        return None

    def _total_orders(df: pd.DataFrame) -> float:
        cart = float(df["cart_orders2"].sum()) if "cart_orders2" in df.columns else 0.0
        phone = float(df["site_phone_orders"].sum()) if "site_phone_orders" in df.columns else 0.0
        return cart + phone

    curr_tot_orders = _total_orders(curr_bucket) if not curr_bucket.empty else 0.0
    prior_tot_orders = _total_orders(prior_bucket) if not prior_bucket.empty else 0.0
    curr_vc_total = _safe_div(curr_tot_orders, curr_tot_sessions)
    prior_vc_total = _safe_div(prior_tot_orders, prior_tot_sessions)

    def _bucket_lookup(df: pd.DataFrame) -> dict[str, dict]:
        if df.empty:
            return {}
        out: dict[str, dict] = {}
        for _, row in df.iterrows():
            b = row["campaign_bucket"]
            sess = float(row.get("sessions", 0.0))
            cart = float(row.get("cart_orders2", 0.0))
            phone = float(row.get("site_phone_orders", 0.0))
            out[b] = {"sessions": sess, "orders": cart + phone}
        return out

    curr_lookup = _bucket_lookup(curr_bucket)
    prior_lookup = _bucket_lookup(prior_bucket)

    all_buckets = sorted(set(curr_lookup).union(prior_lookup))
    ordered = [b for b in CAMPAIGN_BUCKETS if b in all_buckets] + [
        b for b in all_buckets if b not in CAMPAIGN_BUCKETS
    ]

    rows: list[BucketVCDecompRow] = []
    for bucket in ordered:
        curr_sess = curr_lookup.get(bucket, {}).get("sessions", 0.0)
        prior_sess = prior_lookup.get(bucket, {}).get("sessions", 0.0)
        curr_ord = curr_lookup.get(bucket, {}).get("orders", 0.0)
        prior_ord = prior_lookup.get(bucket, {}).get("orders", 0.0)

        if curr_sess == 0 and prior_sess == 0:
            continue

        curr_vc = _safe_div(curr_ord, curr_sess)
        prior_vc = _safe_div(prior_ord, prior_sess)
        curr_w = _safe_div(curr_sess, curr_tot_sessions)
        prior_w = _safe_div(prior_sess, prior_tot_sessions)

        mix_impact = (curr_w - prior_w) * prior_vc
        perf_impact = curr_w * (curr_vc - prior_vc)

        rows.append(BucketVCDecompRow(
            campaign_bucket=bucket,
            curr_sessions=curr_sess,
            prior_sessions=prior_sess,
            curr_orders=curr_ord,
            prior_orders=prior_ord,
            curr_vc=curr_vc,
            prior_vc=prior_vc,
            curr_share=curr_w,
            prior_share=prior_w,
            mix_impact=mix_impact,
            perf_impact=perf_impact,
        ))

    return BucketVCDecompResult(
        rows=rows,
        curr_vc_total=curr_vc_total,
        prior_vc_total=prior_vc_total,
        curr_sessions_total=curr_tot_sessions,
        prior_sessions_total=prior_tot_sessions,
        curr_orders_total=curr_tot_orders,
        prior_orders_total=prior_tot_orders,
    )


# ---------------------------------------------------------------------------
# Drill-down helpers (Section 5)
# ---------------------------------------------------------------------------


def daily_bucket_trend(
    daily_df: pd.DataFrame,
    bucket: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Daily trend for a single bucket over ``[start, end]``, with derived
    rates attached per-day. Used by the bucket drill-down charts.

    Missing days (the bucket had zero activity) are filled with a full
    zero-row so the chart x-axis is continuous.
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    sub = _slice_window(daily_df, start, end)
    sub = sub[sub["campaign_bucket"] == bucket].copy()
    if sub.empty:
        # Fabricate a zero-filled frame so the chart still renders.
        days = pd.date_range(start, end, freq="D").date
        blank = {c: [0.0] * len(days) for c in _NUMERIC_COLUMNS}
        blank["day"] = list(days)
        blank["campaign_bucket"] = [bucket] * len(days)
        return pd.DataFrame(blank)

    sub = sub.sort_values("day").reset_index(drop=True)
    # Reindex so missing days are filled in — saves the caller from having
    # to guard against gaps when building cumulative metrics.
    full_range = pd.date_range(start, end, freq="D").date
    sub = sub.set_index("day").reindex(full_range, fill_value=0.0).rename_axis("day").reset_index()
    sub["campaign_bucket"] = bucket

    # Re-attach derived rates per day (they can't be summed, must be
    # re-derived).
    rate_rows: list[dict] = []
    for _, row in sub.iterrows():
        rate_rows.append(_derived_rates(row.to_dict()))
    return pd.DataFrame(rate_rows)


def bucket_phone_funnel(
    daily_df: pd.DataFrame,
    bucket: str,
    start: date,
    end: date,
) -> dict:
    """
    Per-bucket phone-funnel totals: gross calls → queue calls (broken out
    by lease: grid / homepage / other) → net calls → phone orders, with
    SERP vs non-SERP splits.

    The PRD asks for this view in Section 5. We return a plain dict so
    the Streamlit renderer can format it however it wants (metric cards,
    funnel chart, etc.).
    """
    sub = _slice_window(daily_df, start, end)
    sub = sub[sub["campaign_bucket"] == bucket]
    if sub.empty:
        return {
            "gross_calls": 0.0, "queue_calls": 0.0, "net_calls": 0.0,
            "phone_orders": 0.0, "gross_serp": 0.0, "queue_serp": 0.0,
            "net_serp": 0.0, "serp_orders": 0.0,
            "queue_calls_grid": 0.0, "queue_calls_homepage": 0.0,
            "queue_calls_other": 0.0, "site_queue_calls": 0.0,
            "site_phone_orders": 0.0,
        }
    return {
        "gross_calls": float(sub["gross_calls"].sum()),
        "queue_calls": float(sub["queue_calls"].sum()),
        "net_calls": float(sub["net_calls"].sum()),
        "phone_orders": float(sub["phone_orders"].sum()),
        "gross_serp": float(sub["gross_serp"].sum()),
        "queue_serp": float(sub["queue_serp"].sum()),
        "net_serp": float(sub["net_serp"].sum()),
        "serp_orders": float(sub["serp_orders"].sum()),
        "queue_calls_grid": float(sub["queue_calls_grid"].sum()),
        "queue_calls_homepage": float(sub["queue_calls_homepage"].sum()),
        "queue_calls_other": float(sub["queue_calls_other"].sum()),
        "site_queue_calls": float(sub["site_queue_calls"].sum()),
        "site_phone_orders": float(sub["site_phone_orders"].sum()),
    }


# ---------------------------------------------------------------------------
# Convenience date helpers
# ---------------------------------------------------------------------------


def default_paid_start_date(lookback_days: int = 60) -> date:
    """Reasonable default start date for ad-hoc calls outside the tab."""
    return date.today() - timedelta(days=lookback_days)
