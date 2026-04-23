"""
Organic (SEO) deep-dive tab.

The sidebar channel filter is overridden — every section reports on
`marketing_channel = 'Organic'` (or its GSC analogue). Structure:

  0. TL;DR (mode-aware)                  — synthesized narrative + diagnostic walk
  1. SEO Pacing Summary                  — finance source-of-truth table, SEO only
  2. SEO Revenue Waterfall               — Pacing vs Plan, sequential substitution
  3. Top-of-Funnel Visibility (GSC)      — 4-panel monthly trend chart from d_5
  4. Click Decomposition                 — impression-effect vs CTR-effect + diagnosis
  5. Performance by Landing Page Type    — GSC metrics by site × page_type + unmatched
  6. Top Queries by Page Type            — top-N queries per landing_page_type
  6.5 Top Keyword Ranking Tracker        — top-5 keywords per page_type with
                                            curr vs prior rank + click deltas;
                                            flags |Δrank| ≥ 2 as red and feeds
                                            the TL;DR a structured list of the
                                            biggest keyword movers.
  7. Session Funnel by Page Type         — Organic funnel by landing_page_type

The TL;DR branches on the Organic-window mode:
  • **MoM MTD** — leads with revenue waterfall drivers (vs Plan). Sessions
    driver triggers the flowchart; Cart/Phone drivers get one-line callouts.
  • **WoW** — leads with a 4-metric table (Impressions / Clicks / Sessions /
    Rank) WoW vs prior-7-days vs P4WA; diagnostic flowchart walks against
    WoW deltas.
  • **Custom** — same as WoW but with no P4WA baseline.

Two time systems live on this tab:
  • Sidebar window  — drives Sections 1–2 (Finance / Plan, which are on their
                      own refresh cadence and can safely use yesterday's data).
  • Organic window  — drives Sections 3–7 (GSC + session data). Chosen via a
                      tab-local MoM MTD / WoW / Custom selector, then truncated
                      to the latest fully-reported GSC day so MoM and WoW
                      comparisons are always apples-to-apples.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.app_context import AppContext
from app.finance_data import build_funnel_summary, render_summary_html
from app.narrative import (
    generate_custom_tldr,
    generate_mtd_vs_plan_tldr,
    generate_wow_tldr,
)
from app.seo_diagnostic import DiagnosticReport, walk_diagnostic_tree
from app.seo_data import (
    GSC_SITE_OPTIONS,
    GSC_SITE_TO_DOMAIN,
    aggregate_gsc_daily_to_monthly,
    align_windows_to_gsc,
    bucket_for_landing_page_type,
    compute_click_decomposition,
    diagnose_click_change,
    fetch_gsc_by_page_type,
    fetch_gsc_last_available_date,
    fetch_gsc_p4wa,
    fetch_gsc_page1_churn,
    fetch_gsc_site_trends,
    fetch_gsc_top_keyword_tracker,
    fetch_gsc_top_queries_by_page_type,
    fetch_gsc_unmatched_urls,
    fetch_organic_session_funnel_by_page_type,
)
from app.time_periods import resolve_periods
from app.waterfall import compute_revenue_waterfall, render_waterfall_section


# Consistent palette across all sections (matches the notebook).
_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


# ---------------------------------------------------------------------------
# Date formatting + window-surfacing helpers
# ---------------------------------------------------------------------------
#
# The tab juggles three concurrent time systems (sidebar / Organic / 15-month
# trend) plus a few hardcoded lookback constants. These helpers render each
# one the same way everywhere so the reader can always tell which window a
# section is reporting on.


# Hardcoded inside ``_build_url_to_page_type_cte`` in app/seo_data.py — kept
# here as a constant so the banner can surface it verbatim.
_PAGE_TYPE_LOOKUP_START = "2025-01-01"


def _fmt_date(d: date | None) -> str:
    """Render a date as ``Apr 1, 2026``. Cross-platform safe — avoids the
    ``%-d`` GNU-only flag."""
    if d is None:
        return "—"
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def _window_badge(label: str, start: date | None, end: date | None,
                  extra: str = "") -> None:
    """Render a consistent one-line window caption under a section header.

    Used by every Section 1–7 renderer so the reader never has to scroll
    back to the banner to remember which window drives what. Kept to a
    single st.caption() call for vertical density.
    """
    parts: list[str] = [f"**Window:** {label}"]
    if start is not None and end is not None:
        parts.append(f"{_fmt_date(start)} → {_fmt_date(end)}")
    if extra:
        parts.append(extra)
    st.caption(" · ".join(parts))


def _render_window_banner(ctx: AppContext, period: "OrganicPeriod") -> None:
    """Persistent reference panel shown once under the Organic window picker.

    Lists every date system the tab uses so the reader can scan top-to-bottom
    and tell exactly which window each section is reporting on.
    """
    trend_start = date(period.curr_start.year - 1, 1, 1)
    gsc_through = _fmt_date(period.gsc_max_date) if period.gsc_max_date else "unknown"

    rows = [
        (
            "Finance (sidebar)",
            f"{_fmt_date(ctx.curr_start)} → {_fmt_date(ctx.curr_end)}",
            "Drives Sections 1–2 (Pacing, Waterfall)",
        ),
        (
            "Organic (tab-local)",
            (
                f"{_fmt_date(period.curr_start)} → {_fmt_date(period.curr_end)} "
                f"vs {_fmt_date(period.prior_start)} → {_fmt_date(period.prior_end)}"
            ),
            "Drives Section 0 TL;DR + Sections 3–7 (GSC, Sessions)",
        ),
        (
            "GSC freshness",
            f"data through {gsc_through}",
            "Organic window is truncated to this day for apples-to-apples comparison",
        ),
        (
            "Trend chart (Sec 3)",
            f"{_fmt_date(trend_start)} → {gsc_through}",
            "15-month trend — Jan 1 of prior year through GSC freshness",
        ),
        (
            "Page-type lookup",
            f"sessions since {_PAGE_TYPE_LOOKUP_START}",
            "URL → landing_page_type map used in Sections 5–7",
        ),
    ]

    body_rows = "".join(
        f'<tr>'
        f'<td style="padding:6px 12px;border:1px solid #ddd;'
        f'font-weight:700;white-space:nowrap;">{label}</td>'
        f'<td style="padding:6px 12px;border:1px solid #ddd;'
        f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        f'font-size:13px;">{value}</td>'
        f'<td style="padding:6px 12px;border:1px solid #ddd;'
        f'color:#555;font-size:13px;">{note}</td>'
        f'</tr>'
        for label, value, note in rows
    )
    html = (
        '<table style="border-collapse:collapse;width:100%;'
        'font-family:Arial,sans-serif;font-size:14px;margin-bottom:4px;">'
        '<thead><tr style="background:#2c3e50;color:white;">'
        '<th style="padding:8px 12px;border:1px solid #bbb;text-align:left;">'
        'Time system</th>'
        '<th style="padding:8px 12px;border:1px solid #bbb;text-align:left;">'
        'Window</th>'
        '<th style="padding:8px 12px;border:1px solid #bbb;text-align:left;">'
        'What it drives</th>'
        '</tr></thead>'
        f'<tbody>{body_rows}</tbody>'
        '</table>'
    )
    st.markdown("**Time systems on this tab**")
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Organic period resolution (tab-local — independent of the sidebar)
# ---------------------------------------------------------------------------


@dataclass
class OrganicPeriod:
    """Resolved Organic-tab comparison window, already aligned to GSC freshness."""

    mode: str                       # UI label: "MoM MTD", "WoW", "Custom"
    curr_start: date
    curr_end: date
    prior_start: date
    prior_end: date
    gsc_max_date: date | None
    truncation_note: str            # Human-readable explanation when we clipped windows

    @property
    def curr_days(self) -> int:
        return (self.curr_end - self.curr_start).days + 1

    @property
    def prior_days(self) -> int:
        return (self.prior_end - self.prior_start).days + 1

    @property
    def window_label(self) -> str:
        return (
            f"{self.curr_start.strftime('%b %-d')}–{self.curr_end.strftime('%b %-d')} "
            f"vs {self.prior_start.strftime('%b %-d')}–{self.prior_end.strftime('%b %-d')}"
        )


_TAB_TIME_MODES = ["MoM MTD", "WoW", "Custom"]


def _resolve_tab_period(mode: str, today: date | None = None) -> tuple[date, date, date, date]:
    """Map the tab-local mode label onto `resolve_periods` output."""
    today = today or date.today()
    if mode == "MoM MTD":
        return resolve_periods("MoM", ref_date=today)
    if mode == "WoW":
        return resolve_periods("WoW", ref_date=today)
    if mode == "Custom":
        # Custom mode is handled by the caller via explicit date inputs.
        raise ValueError("Custom mode should use _resolve_tab_period_custom")
    raise ValueError(f"Unknown organic tab time mode: {mode}")


def _render_period_picker() -> tuple[str, tuple[date, date], tuple[date, date]]:
    """Render the tab-local comparison-window picker.

    Returns ``(mode_label, (curr_start, curr_end), (prior_start, prior_end))``
    BEFORE the GSC freshness truncation is applied.
    """
    with st.container():
        cols = st.columns([1, 3])
        with cols[0]:
            mode = st.radio(
                "Organic window",
                options=_TAB_TIME_MODES,
                index=0,
                key="organic_tab_mode",
                help=(
                    "Overrides the sidebar for Sections 3–7 (GSC + session data). "
                    "Finance pacing in Sections 1–2 continues to follow the sidebar."
                ),
            )
        with cols[1]:
            if mode == "Custom":
                today = date.today()
                c1, c2 = st.columns(2)
                with c1:
                    curr_range = st.date_input(
                        "Current",
                        value=(today - timedelta(days=13), today - timedelta(days=7)),
                        key="organic_tab_custom_curr",
                    )
                with c2:
                    prior_range = st.date_input(
                        "Prior",
                        value=(today - timedelta(days=20), today - timedelta(days=14)),
                        key="organic_tab_custom_prior",
                    )
                if not (
                    isinstance(curr_range, tuple) and len(curr_range) == 2
                    and isinstance(prior_range, tuple) and len(prior_range) == 2
                ):
                    st.warning("Pick a start AND end date for both windows.")
                    st.stop()
                curr_start, curr_end = curr_range
                prior_start, prior_end = prior_range
            else:
                curr_start, curr_end, prior_start, prior_end = _resolve_tab_period(mode)
                st.markdown(
                    f"**Current:** `{curr_start} → {curr_end}` ({(curr_end - curr_start).days + 1}d) · "
                    f"**Prior:** `{prior_start} → {prior_end}` ({(prior_end - prior_start).days + 1}d)"
                )

    return mode, (curr_start, curr_end), (prior_start, prior_end)


def _build_organic_period(mode: str, curr: tuple[date, date], prior: tuple[date, date]) -> OrganicPeriod:
    """Run the GSC freshness truncation on the raw picker output."""
    try:
        gsc_max_date = fetch_gsc_last_available_date()
    except Exception:  # pragma: no cover — network/auth failures; degrade gracefully
        gsc_max_date = None

    c_start, c_end, p_start, p_end, note = align_windows_to_gsc(
        curr[0], curr[1], prior[0], prior[1], gsc_max_date
    )
    return OrganicPeriod(
        mode=mode,
        curr_start=c_start,
        curr_end=c_end,
        prior_start=p_start,
        prior_end=p_end,
        gsc_max_date=gsc_max_date,
        truncation_note=note,
    )


# ---------------------------------------------------------------------------
# Shared GSC pulls — so every section reuses the same fetched DataFrames
# instead of re-hitting Databricks for each chart.
# ---------------------------------------------------------------------------


def _site_domain_filter(ui_site: str) -> list[str] | None:
    if ui_site == "All" or ui_site not in GSC_SITE_TO_DOMAIN:
        return None
    return [GSC_SITE_TO_DOMAIN[ui_site]]


def _load_gsc_daily_trends(period: OrganicPeriod, ui_site: str) -> pd.DataFrame:
    """Pull ~15 months of site-level trend data ending at the GSC max date.

    Shared between Section 3 (long-trend chart) and Section 4 (click
    decomposition) so the comparison is guaranteed to use the same data.
    """
    trend_start = date(period.curr_start.year - 1, 1, 1)
    trend_end = period.gsc_max_date or period.curr_end
    domains = _site_domain_filter(ui_site)
    return fetch_gsc_site_trends(str(trend_start), str(trend_end), domains=domains)


def _sum_daily(df: pd.DataFrame, start: date, end: date) -> dict:
    """Aggregate daily GSC rows in a [start, end] window → totals + weighted rank."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    sub = df.loc[mask]
    clicks = float(sub["clicks"].sum()) if not sub.empty else 0.0
    impr = float(sub["impressions"].sum()) if not sub.empty else 0.0
    if impr:
        wrank = float((sub["weighted_avg_rank"] * sub["impressions"]).sum() / impr)
    else:
        wrank = None
    return {"clicks": clicks, "impressions": impr, "weighted_avg_rank": wrank}


# ---------------------------------------------------------------------------
# Sections 1 & 2 — Finance (tied to sidebar, unchanged)
# ---------------------------------------------------------------------------


def _render_sidebar_note() -> None:
    st.info(
        "**Organic deep-dive tab.** Sidebar channel filter is overridden — "
        "every section reports on `marketing_channel = 'Organic'`. Sections 1–2 "
        "(Pacing / Waterfall) follow the **sidebar** time window; the **TL;DR** "
        "at the top and Sections 3–7 (GSC + sessions) follow the tab-local "
        "**Organic window** selector and are truncated to the latest "
        "fully-reported GSC day for apples-to-apples MoM MTD / WoW comparisons."
    )


# ---------------------------------------------------------------------------
# Section 0 — Executive Performance Overview (top landing-page types)
# ---------------------------------------------------------------------------
#
# Two delta-only tables. Top 5 landing-page types by current-period
# organic sessions, aggregated across all 4 GSC properties and grouped
# by the 6-bucket taxonomy (``bucket_for_landing_page_type``):
#
#   Table 1 — Top-of-Funnel Deltas
#     Δ Impressions · Δ Clicks · Δ Sessions · Δ Rank  (PoP only)
#
#   Table 2 — Funnel-Efficiency Deltas
#     Δ Sessions · Δ Phone RR · Δ Phone Conv ·
#     Δ Cart RR · Δ Cart Conversion · Δ Cart VC       (PoP only)
#
# No raw values are shown — GSC is heavily censored and the PoP delta
# is the honest signal because the same censoring applies on both sides.
# The PoP label in the header is mode-aware: MoM / WoW / PoP.
#
# Conditional formatting:
#   green  — improved by  > +2.5% (or rank better by > 0.5 positions)
#   red    — worsened by  < -2.5% (or rank worse  by > 0.5 positions)
#   yellow — within the stability band
#   gray   — no data / divide-by-zero
#
# Tier{1-4}CityGEO rows are collapsed into a single `TierCityGEO` bucket
# and all electricity-plan templates collapse into `PlanType` per the
# exec spec — these templates are interchangeable for strategic
# purposes, so breaking them out adds noise.


def _pct_change(curr: float | None, prior: float | None) -> float | None:
    """Fractional change. Returns None on missing inputs or zero prior."""
    if curr is None or prior is None or pd.isna(curr) or pd.isna(prior) or prior == 0:
        return None
    return (curr - prior) / prior


# The exec-view bucket taxonomy lives in ``app/seo_data.py``
# (``bucket_for_landing_page_type``). Previously this module defined its
# own partial rollup (TierCityGEO + PlanType only); the canonical helper
# now handles the full 6-bucket taxonomy — Homepage, StateGEO, CityGEO,
# Provider, PlanType, Informational — plus the ``Unmatched`` hygiene
# bucket. The ``_collapse_*`` names below are kept as thin aliases so any
# import path that still references them keeps working without changes.
_collapse_landing_page_type = bucket_for_landing_page_type
_collapse_tier_city_geo = bucket_for_landing_page_type


def _rollup_page_type_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse landing_page_type values via ``_collapse_landing_page_type``
    (TierCityGEO + PlanType rollups), then sum metrics across sites.

    Input columns (from ``fetch_gsc_by_page_type``):
      site, domain, landing_page_type, clicks, impressions, ctr,
      weighted_avg_rank

    Output: one row per landing_page_type (cross-site) with the totals and
    impression-weighted rank / CTR. ``ctr`` and ``weighted_avg_rank`` in
    the input are recomputed from the raw numerators — the input aggregate
    rates aren't directly summable.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "landing_page_type", "clicks", "impressions", "ctr",
            "weighted_avg_rank",
        ])
    tmp = df.copy()
    tmp["landing_page_type"] = tmp["landing_page_type"].map(_collapse_landing_page_type)
    tmp["pos_x_impr"] = tmp["weighted_avg_rank"].fillna(0) * tmp["impressions"].fillna(0)

    agg = (
        tmp.groupby("landing_page_type", as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            pos_x_impr=("pos_x_impr", "sum"),
        )
    )
    agg["ctr"] = agg["clicks"] / agg["impressions"].replace(0, pd.NA)
    agg["weighted_avg_rank"] = agg["pos_x_impr"] / agg["impressions"].replace(0, pd.NA)
    return agg[[
        "landing_page_type", "clicks", "impressions", "ctr", "weighted_avg_rank",
    ]]


def _rollup_session_funnel_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sum session-funnel primitives (sessions, queue_calls, phone_orders,
    carts, cart_orders) by ``landing_page_type`` across sites, applying the
    exec-view rollup, and recompute the per-bucket rates from the raw
    totals.

    Used for the delta-only executive tables where we need both top-of-
    funnel (sessions) and bottom-of-funnel (Phone RR / VC, Cart RR /
    Conversion / VC) from a single page-type × window pull.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "landing_page_type", "sessions", "queue_calls", "phone_orders",
            "carts", "cart_orders",
            "phone_rr_pct", "phone_vc_pct",
            "cart_rate_pct", "cart_conversion_pct", "cart_vc_pct",
        ])
    tmp = df.copy()
    tmp["landing_page_type"] = tmp["landing_page_type"].map(_collapse_landing_page_type)
    sum_cols = [
        c for c in ["sessions", "queue_calls", "phone_orders",
                    "carts", "cart_orders"] if c in tmp.columns
    ]
    agg = tmp.groupby("landing_page_type", as_index=False)[sum_cols].sum()
    safe_sessions = agg["sessions"].replace(0, pd.NA) if "sessions" in agg.columns else None
    if safe_sessions is not None:
        if "queue_calls" in agg.columns:
            agg["phone_rr_pct"] = agg["queue_calls"] / safe_sessions
        if "phone_orders" in agg.columns:
            agg["phone_vc_pct"] = agg["phone_orders"] / safe_sessions
        if "carts" in agg.columns:
            agg["cart_rate_pct"] = agg["carts"] / safe_sessions
        if "cart_orders" in agg.columns:
            agg["cart_vc_pct"] = agg["cart_orders"] / safe_sessions
    if "cart_orders" in agg.columns and "carts" in agg.columns:
        agg["cart_conversion_pct"] = agg["cart_orders"] / agg["carts"].replace(0, pd.NA)
    return agg


def _compute_landing_page_type_exec_table(
    period: OrganicPeriod,
    ui_site: str = "All",
) -> pd.DataFrame | None:
    """Build the exec-view landing-page-type delta frame.

    Pulls four datasets, all rolled up to the 6-bucket taxonomy
    (``bucket_for_landing_page_type``):
      • GSC page-type for the current window (``fetch_gsc_by_page_type``)
      • GSC page-type for the prior window (same function, prior dates)
      • Session funnel for the current window
      • Session funnel for the prior window

    ``ui_site`` controls the brand scope. Values are the same strings
    surfaced by the tab's Site filter radio — ``"All"``, ``"CTXP"``,
    ``"SOE"``, ``"Choose TX"``, ``"TXER"``. When not ``"All"`` we
    restrict **both** the GSC fetch (via its domain list) **and** the
    session-funnel fetch (via its ``websites`` filter) so the two sides
    of the comparison come from the same brand.

    The output has one row per bucket and carries all current + prior
    values plus the PoP deltas — callers render **deltas only**. GSC
    data is notoriously incomplete (a huge share of impressions/clicks/
    queries are anonymized), so displaying raw values invites over-
    interpretation. The PoP delta is much more stable because the same
    censoring applies on both sides of the comparison.

    Rows are ranked by current-period Organic sessions and the top 5
    are returned. Buckets with minimal current traffic (< 1k impressions)
    and the ``Unmatched`` hygiene bucket are dropped before ranking.
    """
    domain_list = _site_domain_filter(ui_site)  # None for "All"
    website_filter: tuple[str, ...] | None = (
        (ui_site,) if ui_site != "All" and ui_site in GSC_SITE_TO_DOMAIN else None
    )

    curr_df_raw = fetch_gsc_by_page_type(
        str(period.curr_start), str(period.curr_end), domains=domain_list,
    )
    prior_df_raw = fetch_gsc_by_page_type(
        str(period.prior_start), str(period.prior_end), domains=domain_list,
    )
    curr_pt = _rollup_page_type_frame(curr_df_raw)
    prior_pt = _rollup_page_type_frame(prior_df_raw)
    if curr_pt.empty:
        return None

    # Session funnel — curr + prior, same bucket taxonomy.
    try:
        curr_sess_raw = fetch_organic_session_funnel_by_page_type(
            str(period.curr_start), str(period.curr_end),
            websites=website_filter,
        )
        curr_sess = _rollup_session_funnel_frame(curr_sess_raw)
    except Exception:
        curr_sess = pd.DataFrame(columns=["landing_page_type"])
    try:
        prior_sess_raw = fetch_organic_session_funnel_by_page_type(
            str(period.prior_start), str(period.prior_end),
            websites=website_filter,
        )
        prior_sess = _rollup_session_funnel_frame(prior_sess_raw)
    except Exception:
        prior_sess = pd.DataFrame(columns=["landing_page_type"])

    # Merge GSC curr/prior.
    base = curr_pt.rename(columns={
        "clicks": "clicks_curr", "impressions": "impressions_curr",
        "ctr": "ctr_curr", "weighted_avg_rank": "rank_curr",
    })
    p = prior_pt.rename(columns={
        "clicks": "clicks_prior", "impressions": "impressions_prior",
        "ctr": "ctr_prior", "weighted_avg_rank": "rank_prior",
    })
    frame = base.merge(p, on="landing_page_type", how="left")

    # Merge session funnel curr/prior.
    sess_rate_cols = [
        "sessions", "phone_rr_pct", "phone_vc_pct",
        "cart_rate_pct", "cart_conversion_pct", "cart_vc_pct",
    ]
    curr_keep = ["landing_page_type"] + [c for c in sess_rate_cols if c in curr_sess.columns]
    prior_keep = ["landing_page_type"] + [c for c in sess_rate_cols if c in prior_sess.columns]
    curr_sess_ren = curr_sess[curr_keep].rename(columns={
        c: f"{c}_curr" for c in curr_keep if c != "landing_page_type"
    })
    prior_sess_ren = prior_sess[prior_keep].rename(columns={
        c: f"{c}_prior" for c in prior_keep if c != "landing_page_type"
    })
    frame = frame.merge(curr_sess_ren, on="landing_page_type", how="left")
    frame = frame.merge(prior_sess_ren, on="landing_page_type", how="left")

    frame["sessions_curr"] = pd.to_numeric(
        frame.get("sessions_curr"), errors="coerce"
    ).fillna(0)
    frame["sessions_prior"] = pd.to_numeric(
        frame.get("sessions_prior"), errors="coerce"
    ).fillna(0)

    # Drop noise: Unmatched + tiny buckets. 1k impressions is a sensible
    # floor for the all-brand cut (~15k impressions per brand in a typical
    # week). Relax to 200 when a single brand is selected so smaller sites
    # (TXER, Choose TX) still surface their top buckets.
    impression_floor = 1_000 if ui_site == "All" else 200
    frame = frame[
        (frame["landing_page_type"] != "Unmatched")
        & (frame["impressions_curr"].fillna(0) >= impression_floor)
    ]
    if frame.empty:
        return None

    frame = frame.sort_values(
        ["sessions_curr", "clicks_curr"], ascending=[False, False]
    ).head(5).reset_index(drop=True)

    # Top-of-funnel deltas (% for volumes, positional for rank).
    for base_name in ["impressions", "clicks", "sessions"]:
        frame[f"{base_name}_pop_delta"] = frame.apply(
            lambda r, b=base_name: _pct_change(r[f"{b}_curr"], r[f"{b}_prior"]),
            axis=1,
        )
    frame["rank_pop_delta"] = frame["rank_curr"] - frame["rank_prior"]

    # Funnel-efficiency deltas (% change in rates).
    for rate_name in ["phone_rr_pct", "phone_vc_pct", "cart_rate_pct",
                      "cart_conversion_pct", "cart_vc_pct"]:
        curr_col = f"{rate_name}_curr"
        prior_col = f"{rate_name}_prior"
        if curr_col in frame.columns and prior_col in frame.columns:
            frame[f"{rate_name}_pop_delta"] = frame.apply(
                lambda r, c=curr_col, p_=prior_col: _pct_change(r[c], r[p_]),
                axis=1,
            )
        else:
            frame[f"{rate_name}_pop_delta"] = pd.NA

    return frame


# ---------------------------------------------------------------------------
# Exec tables are rendered as raw HTML (not st.dataframe / Pandas Styler) so
# that copy-paste into Slack, Slides, or email preserves the styling. The
# look mirrors `app/finance_data.render_summary_html`:
#
#   • Dark navy header (#2c3e50) with white text, centered.
#   • 1px solid borders on every cell (#bbb header, #ddd body).
#   • Excel conditional-formatting palette for delta cells — the same
#     green / red / yellow palette execs already recognize from the
#     finance pacing table.
#   • Bold numeric values so they stand out against the colored cells.
# ---------------------------------------------------------------------------

# Stability bands — below these magnitudes a delta is treated as "flat"
# (yellow) rather than green or red.
_EXEC_PCT_THRESHOLD = 0.025         # ±2.5% band for %-change metrics
_EXEC_RANK_THRESHOLD = 0.5          # ±0.5 position band for rank deltas

# Excel-style conditional-formatting palette — exact match to
# `_delta_style()` in finance_data.py so the two report families look
# consistent side-by-side.
_EXEC_COLOR_GREEN = "background-color:#c6efce;color:#006100;"
_EXEC_COLOR_RED = "background-color:#ffc7ce;color:#9c0006;"
_EXEC_COLOR_YELLOW = "background-color:#ffeb9c;color:#9c6500;"
_EXEC_COLOR_NEUTRAL = ""

# Shared table chrome (reused by both `_render_top_of_funnel_delta_table`
# and `_render_funnel_efficiency_delta_table`).
_EXEC_TABLE_STYLE = (
    "border-collapse:collapse;width:100%;font-family:Arial,sans-serif;"
    "font-size:14px;"
)
_EXEC_HEADER_ROW_STYLE = (
    "background:#2c3e50;color:white;text-align:center;"
)
_EXEC_HEADER_CELL_STYLE = "padding:10px 12px;border:1px solid #bbb;font-weight:700;"
_EXEC_BODY_CELL_STYLE = (
    "padding:8px 12px;border:1px solid #ddd;text-align:center;font-weight:600;"
)
_EXEC_BODY_LABEL_STYLE = (
    "padding:8px 14px;border:1px solid #ddd;font-weight:700;text-align:left;"
)


def _fmt_int_or_dash(x) -> str:
    return f"{x:,.0f}" if pd.notna(x) else "—"


def _fmt_pct_change(x) -> str:
    return f"{x * 100:+.1f}%" if pd.notna(x) else "—"


def _fmt_pct_value(x) -> str:
    return f"{x * 100:.2f}%" if pd.notna(x) else "—"


def _fmt_rank_value(x) -> str:
    return f"{x:.1f}" if pd.notna(x) else "—"


def _fmt_rank_delta(x) -> str:
    return f"{x:+.2f}" if pd.notna(x) else "—"


def _exec_delta_style(value, *, kind: str = "pct", inverted: bool = False) -> str:
    """Return the inline CSS for a delta cell.

    Parameters
    ----------
    value : number or NaN
        Raw delta. Fractional for %-change (``kind='pct'``),
        positional for rank (``kind='rank'``).
    kind : {'pct', 'rank'}
        Chooses the stability threshold (``_EXEC_PCT_THRESHOLD`` vs.
        ``_EXEC_RANK_THRESHOLD``).
    inverted : bool
        ``True`` for rank cells — a negative positional delta is *good*
        (rank went from 7 → 5), so the sign-check flips.
    """
    if value is None or pd.isna(value):
        return _EXEC_COLOR_NEUTRAL
    threshold = _EXEC_RANK_THRESHOLD if kind == "rank" else _EXEC_PCT_THRESHOLD
    if abs(value) < threshold:
        return _EXEC_COLOR_YELLOW
    positive = (value > 0) != inverted
    return _EXEC_COLOR_GREEN if positive else _EXEC_COLOR_RED


def _exec_table_header(labels: list[str]) -> str:
    """Build the navy-background header row for an exec table.

    The first header cell (Page Type) is left-aligned; every other is
    centered to match the Excel reference design.
    """
    cells = []
    for i, lbl in enumerate(labels):
        align = "text-align:left;min-width:140px;" if i == 0 else ""
        cells.append(f'<th style="{_EXEC_HEADER_CELL_STYLE}{align}">{lbl}</th>')
    return (
        f'<thead><tr style="{_EXEC_HEADER_ROW_STYLE}">'
        f'{"".join(cells)}</tr></thead>'
    )


def _exec_table_row(cells: list[tuple[str, str]]) -> str:
    """Build one body row. ``cells`` is a list of ``(value_html, extra_css)``
    tuples. The first cell gets the page-type label style; the rest get
    the numeric-body style merged with any per-cell delta coloring.
    """
    bits = []
    for i, (val, extra) in enumerate(cells):
        if i == 0:
            bits.append(f'<td style="{_EXEC_BODY_LABEL_STYLE}{extra}">{val}</td>')
        else:
            bits.append(f'<td style="{_EXEC_BODY_CELL_STYLE}{extra}">{val}</td>')
    return "<tr>" + "".join(bits) + "</tr>"


def _pop_label(period: OrganicPeriod) -> str:
    """Short comparison label for the column headers. Mode-aware:
    ``MoM`` for MTD, ``WoW`` for the weekly picker, ``PoP`` otherwise."""
    if period.mode == "MoM MTD":
        return "MoM"
    if period.mode == "WoW":
        return "WoW"
    return "PoP"


def _render_top_of_funnel_delta_table(
    frame: pd.DataFrame, period: OrganicPeriod,
) -> None:
    """Section 0 / Table 1 — **Top-of-Funnel Deltas**.

    ``Page Type | Δ Impressions | Δ Clicks | Δ Sessions | Δ Rank``.

    Deltas only — no raw values. GSC impressions, clicks, and rank are
    subject to substantial censoring (anonymized queries, thresholds),
    so raw counts overstate precision. The PoP comparison holds the
    censoring roughly constant on both sides, which is why the **change**
    is the useful signal.

    Rank Δ is positional (``curr - prior``) and inverted in the color
    map — rank 7 → 5 is a good move (negative delta → green).
    """
    pop = _pop_label(period)
    header = [
        "Page Type",
        f"Δ Impressions ({pop})",
        f"Δ Clicks ({pop})",
        f"Δ Sessions ({pop})",
        f"Δ Rank ({pop})",
    ]

    rows_html: list[str] = []
    for _, row in frame.iterrows():
        cells: list[tuple[str, str]] = [
            (str(row["landing_page_type"]), ""),
            (
                _fmt_pct_change(row["impressions_pop_delta"]),
                _exec_delta_style(row["impressions_pop_delta"], kind="pct"),
            ),
            (
                _fmt_pct_change(row["clicks_pop_delta"]),
                _exec_delta_style(row["clicks_pop_delta"], kind="pct"),
            ),
            (
                _fmt_pct_change(row["sessions_pop_delta"]),
                _exec_delta_style(row["sessions_pop_delta"], kind="pct"),
            ),
            (
                _fmt_rank_delta(row["rank_pop_delta"]),
                _exec_delta_style(row["rank_pop_delta"], kind="rank", inverted=True),
            ),
        ]
        rows_html.append(_exec_table_row(cells))

    html = (
        f'<table style="{_EXEC_TABLE_STYLE}">'
        f"{_exec_table_header(header)}"
        f'<tbody>{"".join(rows_html)}</tbody>'
        "</table>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_funnel_efficiency_delta_table(
    frame: pd.DataFrame, period: OrganicPeriod,
) -> None:
    """Section 0 / Table 2 — **Funnel-Efficiency Deltas**.

    ``Page Type | Δ Sessions | Δ Phone RR | Δ Phone Conv |
      Δ Cart RR | Δ Cart Conv | Δ Cart VC``.

    Deltas only. Phone RR = queue_calls / sessions;
    Phone Conv (aka Phone VC) = phone_orders / sessions;
    Cart RR = carts / sessions;
    Cart Conversion = cart_orders / carts;
    Cart VC = cart_orders / sessions. Each Δ is the % change in the
    per-bucket rate between the current and prior window.
    """
    pop = _pop_label(period)
    header = [
        "Page Type",
        f"Δ Sessions ({pop})",
        f"Δ Phone RR ({pop})",
        f"Δ Phone Conv ({pop})",
        f"Δ Cart RR ({pop})",
        f"Δ Cart Conv ({pop})",
        f"Δ Cart VC ({pop})",
    ]

    rows_html: list[str] = []
    for _, row in frame.iterrows():
        cells: list[tuple[str, str]] = [
            (str(row["landing_page_type"]), ""),
            (
                _fmt_pct_change(row["sessions_pop_delta"]),
                _exec_delta_style(row["sessions_pop_delta"], kind="pct"),
            ),
            (
                _fmt_pct_change(row.get("phone_rr_pct_pop_delta")),
                _exec_delta_style(row.get("phone_rr_pct_pop_delta"), kind="pct"),
            ),
            (
                _fmt_pct_change(row.get("phone_vc_pct_pop_delta")),
                _exec_delta_style(row.get("phone_vc_pct_pop_delta"), kind="pct"),
            ),
            (
                _fmt_pct_change(row.get("cart_rate_pct_pop_delta")),
                _exec_delta_style(row.get("cart_rate_pct_pop_delta"), kind="pct"),
            ),
            (
                _fmt_pct_change(row.get("cart_conversion_pct_pop_delta")),
                _exec_delta_style(row.get("cart_conversion_pct_pop_delta"), kind="pct"),
            ),
            (
                _fmt_pct_change(row.get("cart_vc_pct_pop_delta")),
                _exec_delta_style(row.get("cart_vc_pct_pop_delta"), kind="pct"),
            ),
        ]
        rows_html.append(_exec_table_row(cells))

    html = (
        f'<table style="{_EXEC_TABLE_STYLE}">'
        f"{_exec_table_header(header)}"
        f'<tbody>{"".join(rows_html)}</tbody>'
        "</table>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_exec_overview(
    period: OrganicPeriod,
    organic_sessions: dict | None,
    ui_site: str = "All",
) -> None:
    """Section 0 — two compact delta-only tables covering the top 5
    landing-page types by current-period Organic sessions.

    Table 1 (Top-of-Funnel)     — Δ Impressions / Δ Clicks / Δ Sessions / Δ Rank.
    Table 2 (Funnel Efficiency) — Δ Sessions / Δ Phone RR / Δ Phone Conv /
                                   Δ Cart RR / Δ Cart Conv / Δ Cart VC.

    Rationale: GSC-side counts (impressions, clicks, rank samples) are
    heavily censored, so raw levels invite over-reading of noise. The PoP
    delta holds the censoring roughly constant on both sides and is the
    honest signal. Rates in Table 2 come from the session-level funnel
    and give the "did we convert traffic better?" read in one glance.

    Scope is controlled by ``ui_site``: ``"All"`` aggregates across all
    4 GSC properties, any other value restricts **both** the GSC pull
    (by domain) and the session-funnel pull (by ``website``). The
    ``TierCityGEO`` and ``PlanType`` rollups collapse the sub-tier and
    electricity-plan templates so execs see strategic buckets, not
    template-level noise.
    """
    st.header("Executive Performance Overview")

    pop = _pop_label(period)
    site_scope = (
        "aggregated across all 4 Texas brands"
        if ui_site == "All"
        else f"scoped to **{ui_site}** only"
    )
    st.caption(
        f"Top 5 landing-page types by current-period Organic sessions, "
        f"{site_scope}. "
        f"**{period.mode}** ({period.curr_start} → {period.curr_end}) "
        f"vs **{period.prior_start} → {period.prior_end}**. "
        f"All columns are **{pop} deltas only** — raw GSC counts are "
        f"censored and over-state precision. "
        f"Rollups: **TierCityGEO** = Tier1–4 CityGEO; "
        f"**PlanType** = Solar + all `*_Plans`. "
        f"Green = improved >+2.5% (or rank better >0.5), "
        f"red = worse by the same threshold, yellow = within band, gray = n/a."
    )
    if period.truncation_note:
        st.caption(f":information_source: {period.truncation_note}")

    with st.spinner("Building top-5 landing-page-type overview…"):
        try:
            frame = _compute_landing_page_type_exec_table(period, ui_site)
        except Exception as e:  # pragma: no cover
            st.error(f"Couldn't build the landing-page-type table: {e}")
            return

    if frame is None or frame.empty:
        st.info(
            "No landing-page-type data available for the current window "
            "(GSC impressions below threshold or session data missing)."
        )
        return

    st.markdown("**Top-of-Funnel — are we reaching and converting search demand?**")
    _render_top_of_funnel_delta_table(frame, period)

    st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical spacer between tables
    st.markdown("**Funnel Efficiency — once on-site, are we converting sessions?**")
    _render_funnel_efficiency_delta_table(frame, period)


def _organic_window_badge(period: "OrganicPeriod", *, extra: str = "",
                          current_only: bool = False) -> None:
    """Shortcut for the Organic (tab-local) window badge used in Sections 4–7.

    Shows both current and prior windows in the same format as
    ``_window_badge`` so the reader can diff them at a glance. Passing
    ``current_only=True`` renders just the current window (used by the
    session-funnel section which is current-only).
    """
    if current_only:
        _window_badge(
            "Organic (current only)", period.curr_start, period.curr_end,
            extra=extra,
        )
        return
    label = (
        f"Organic · {_fmt_date(period.curr_start)} → {_fmt_date(period.curr_end)} "
        f"vs {_fmt_date(period.prior_start)} → {_fmt_date(period.prior_end)}"
    )
    parts = [f"**Window:** {label}"]
    if extra:
        parts.append(extra)
    st.caption(" · ".join(parts))


def _render_pacing_summary(ctx: AppContext) -> None:
    st.header("1. SEO Pacing Summary")
    _window_badge("Sidebar (Finance)", ctx.curr_start, ctx.curr_end)
    st.caption("Sidebar-driven. Source: finance pacing (`rpt_texas_daily_pacing`).")
    if ctx.finance_df is None or ctx.plan_df is None:
        st.warning("Finance pacing data unavailable.")
        return

    try:
        rows = build_funnel_summary(
            ctx.finance_df,
            ctx.plan_df,
            ["Organic"],
            curr_start=ctx.curr_start,
            curr_end=ctx.curr_end,
            prior_start=ctx.prior_start,
            prior_end=ctx.prior_end,
        )
        st.markdown(
            render_summary_html(rows, date.today(), period_label=ctx.period_label),
            unsafe_allow_html=True,
        )
    except Exception as e:  # pragma: no cover
        st.warning(f"SEO pacing summary unavailable: {e}")


def _render_waterfall(ctx: AppContext) -> None:
    st.header("2. SEO Revenue Waterfall")
    _window_badge("Sidebar (Finance)", ctx.curr_start, ctx.curr_end)
    st.caption(
        "Sidebar-driven. Pacing vs Plan revenue gap for Organic only, decomposed "
        "via sequential substitution across Sessions → Site RR → Site Conversion "
        "→ Phone GCV/Order → Cart RR → Cart Conversion → Cart GCV/Order."
    )
    result = render_waterfall_section(
        ctx.plan_df,
        ["Organic"],
        chart_title="SEO Revenue Waterfall — Plan → Pacing",
        caption="Source: `rpt_texas_daily_pacing` filtered to Organic + SEO plan channels.",
    )
    if result is not None:
        ctx.cache["organic_waterfall"] = result


# ---------------------------------------------------------------------------
# Section 0 — Mode-aware TL;DR
# ---------------------------------------------------------------------------


def _compute_organic_sessions_delta(
    ctx: AppContext, period: OrganicPeriod
) -> dict | None:
    """Organic session count for the current + prior windows, from finance.

    Used by the diagnostic walker's trigger and by the MTD TL;DR's headline.
    """
    if ctx.finance_df is None or "MarketingChannel" not in ctx.finance_df.columns:
        return None
    fin = ctx.finance_df[ctx.finance_df["MarketingChannel"] == "Organic"]
    if fin.empty or "TheDate" not in fin.columns or "Total_Sessions" not in fin.columns:
        return None

    def _sum(start: date, end: date) -> float:
        sub = fin[(fin["TheDate"] >= start) & (fin["TheDate"] <= end)]
        return float(sub["Total_Sessions"].sum())

    curr = _sum(period.curr_start, period.curr_end)
    prior = _sum(period.prior_start, period.prior_end)
    if prior == 0:
        return {"curr_sessions": curr, "prior_sessions": prior, "pct_change": None, "delta": curr - prior}
    return {
        "curr_sessions": curr,
        "prior_sessions": prior,
        "pct_change": (curr - prior) / prior,
        "delta": curr - prior,
    }


def _fetch_page1_churn_safe(
    period: OrganicPeriod, ui_site: str
) -> dict:
    """Wrap `fetch_gsc_page1_churn` so a failure doesn't kill the TL;DR."""
    domains = _site_domain_filter(ui_site)
    try:
        return fetch_gsc_page1_churn(
            curr_start=period.curr_start,
            curr_end=period.curr_end,
            prior_start=period.prior_start,
            prior_end=period.prior_end,
            domains=domains,
        )
    except Exception:  # pragma: no cover — network/auth degradation
        return {"churn_pct": None, "prior_page1_queries": 0,
                "churned_queries": 0, "examples": []}


def _fetch_p4wa_safe(period: OrganicPeriod, ui_site: str) -> dict:
    """Wrap `fetch_gsc_p4wa` so a failure doesn't kill the TL;DR."""
    domains = _site_domain_filter(ui_site)
    try:
        return fetch_gsc_p4wa(
            period.curr_start, period.curr_end, domains=domains,
        )
    except Exception:  # pragma: no cover
        return {}


def _build_4metric_table(
    *,
    decomp: dict | None,
    curr_rank: float | None,
    prior_rank: float | None,
    organic_sessions: dict | None,
    p4wa: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Assemble the 4-metric compact table for the TL;DR header.

    Returns ``(llm_payload_dict, dataframe)`` where the DataFrame is the
    display form (already formatted strings) and the dict is the structured
    form fed to the LLM context.
    """
    rows: list[dict] = []
    llm_payload: dict = {}

    def _fmt(value: float | None, *, kind: str) -> str:
        if value is None:
            return "—"
        if kind == "int":
            return f"{int(value):,}"
        if kind == "rank":
            return f"{value:.1f}"
        if kind == "pct":
            return f"{value * 100:.2f}%"
        return str(value)

    def _fmt_delta(curr: float | None, prior: float | None,
                   *, kind: str) -> tuple[str, str]:
        if curr is None or prior is None:
            return "—", "—"
        diff = curr - prior
        if kind == "rank":
            return f"{diff:+.2f}", "—"
        if kind == "pct":
            return f"{diff * 100:+.2f}pp", (
                f"{(diff / prior) * 100:+.1f}%" if prior else "—"
            )
        # int
        return f"{diff:+,.0f}", (f"{(diff / prior) * 100:+.1f}%" if prior else "—")

    def _p4wa_delta(curr: float | None, p4wa_val: float | None,
                    *, kind: str) -> str:
        if curr is None or p4wa_val is None or p4wa_val == 0:
            return "—"
        if kind == "rank":
            return f"{curr - p4wa_val:+.2f}"
        return f"{(curr - p4wa_val) / p4wa_val * 100:+.1f}%"

    # ── Impressions ───────────────────────────────────────────────────
    curr_impr = (decomp or {}).get("curr_impressions")
    prior_impr = (decomp or {}).get("prior_impressions")
    p4wa_impr = (p4wa or {}).get("impressions")
    delta_abs, delta_pct = _fmt_delta(curr_impr, prior_impr, kind="int")
    p4wa_delta = _p4wa_delta(curr_impr, p4wa_impr, kind="int")
    rows.append({
        "Metric": "Impressions",
        "Current": _fmt(curr_impr, kind="int"),
        "Prior": _fmt(prior_impr, kind="int"),
        "Δ": delta_abs,
        "Δ%": delta_pct,
        "P4WA": _fmt(p4wa_impr, kind="int"),
        "Δ% vs P4WA": p4wa_delta,
    })
    llm_payload["impressions"] = {
        "curr": _fmt(curr_impr, kind="int"),
        "prior": _fmt(prior_impr, kind="int"),
        "delta": delta_abs,
        "pct_change": delta_pct,
        "p4wa": _fmt(p4wa_impr, kind="int") if p4wa_impr else None,
        "pct_change_vs_p4wa": p4wa_delta if p4wa_impr else None,
    }

    # ── Clicks ────────────────────────────────────────────────────────
    curr_clk = (decomp or {}).get("curr_clicks")
    prior_clk = (decomp or {}).get("prior_clicks")
    p4wa_clk = (p4wa or {}).get("clicks")
    delta_abs, delta_pct = _fmt_delta(curr_clk, prior_clk, kind="int")
    p4wa_delta = _p4wa_delta(curr_clk, p4wa_clk, kind="int")
    rows.append({
        "Metric": "Clicks",
        "Current": _fmt(curr_clk, kind="int"),
        "Prior": _fmt(prior_clk, kind="int"),
        "Δ": delta_abs,
        "Δ%": delta_pct,
        "P4WA": _fmt(p4wa_clk, kind="int"),
        "Δ% vs P4WA": p4wa_delta,
    })
    llm_payload["clicks"] = {
        "curr": _fmt(curr_clk, kind="int"),
        "prior": _fmt(prior_clk, kind="int"),
        "delta": delta_abs,
        "pct_change": delta_pct,
        "p4wa": _fmt(p4wa_clk, kind="int") if p4wa_clk else None,
        "pct_change_vs_p4wa": p4wa_delta if p4wa_clk else None,
    }

    # ── Sessions ──────────────────────────────────────────────────────
    curr_sess = (organic_sessions or {}).get("curr_sessions")
    prior_sess = (organic_sessions or {}).get("prior_sessions")
    delta_abs, delta_pct = _fmt_delta(curr_sess, prior_sess, kind="int")
    rows.append({
        "Metric": "Sessions",
        "Current": _fmt(curr_sess, kind="int"),
        "Prior": _fmt(prior_sess, kind="int"),
        "Δ": delta_abs,
        "Δ%": delta_pct,
        "P4WA": "—",
        "Δ% vs P4WA": "—",
    })
    llm_payload["sessions"] = {
        "curr": _fmt(curr_sess, kind="int"),
        "prior": _fmt(prior_sess, kind="int"),
        "delta": delta_abs,
        "pct_change": delta_pct,
    }

    # ── Weighted avg rank (inverse: lower is better) ──────────────────
    p4wa_rank = (p4wa or {}).get("weighted_avg_rank")
    delta_abs, _ = _fmt_delta(curr_rank, prior_rank, kind="rank")
    p4wa_delta = _p4wa_delta(curr_rank, p4wa_rank, kind="rank")
    rows.append({
        "Metric": "Weighted avg rank",
        "Current": _fmt(curr_rank, kind="rank"),
        "Prior": _fmt(prior_rank, kind="rank"),
        "Δ": delta_abs,
        "Δ%": "—",
        "P4WA": _fmt(p4wa_rank, kind="rank"),
        "Δ% vs P4WA": p4wa_delta,
    })
    llm_payload["rank"] = {
        "curr": _fmt(curr_rank, kind="rank"),
        "prior": _fmt(prior_rank, kind="rank"),
        "delta": delta_abs,
        "p4wa": _fmt(p4wa_rank, kind="rank") if p4wa_rank else None,
        "pct_change_vs_p4wa": p4wa_delta if p4wa_rank else None,
    }

    return llm_payload, pd.DataFrame(rows)


def _run_flowchart(
    *,
    period: OrganicPeriod,
    decomp: dict | None,
    curr_rank: float | None,
    prior_rank: float | None,
    page_type_top_movers: list[dict],
    organic_sessions: dict | None,
    page1_churn: dict,
    session_share_map: dict[str, float] | None = None,
) -> DiagnosticReport:
    """Invoke the flowchart walker with the full set of signals."""
    return walk_diagnostic_tree(
        session_curr=(organic_sessions or {}).get("curr_sessions"),
        session_prior=(organic_sessions or {}).get("prior_sessions"),
        impression_effect=(decomp or {}).get("impression_effect"),
        ctr_effect=(decomp or {}).get("ctr_effect"),
        pct_change_impressions=(decomp or {}).get("pct_change_impressions"),
        pct_change_ctr=(decomp or {}).get("pct_change_ctr"),
        curr_rank=curr_rank,
        prior_rank=prior_rank,
        page_type_movers=page_type_top_movers,
        page1_churn_pct=page1_churn.get("churn_pct"),
        window_label=period.window_label,
        session_share_map=session_share_map,
    )


def _build_waterfall_payload(ctx: AppContext) -> dict | None:
    """Compute the Organic revenue waterfall and package the top drivers
    in the structure the MTD TL;DR expects.
    """
    if ctx.plan_df is None:
        return None
    result = compute_revenue_waterfall(ctx.plan_df, ["Organic"])
    if result is None:
        return None

    impacts = result.impacts  # [(name, dollar), ...]
    gap = result.total_gap or 1.0
    plan = result.plan
    actual = result.actual

    def _plan_pacing_formatted(name: str) -> tuple[str, str, str]:
        """Return (plan_fmt, pacing_fmt, vs_plan_pct) for a driver."""
        mapping = {
            "Sessions":         (plan.sessions,         actual.sessions,         "num"),
            "Site RR":          (plan.site_rr,          actual.site_rr,          "rate"),
            "Site Conversion":  (plan.site_conversion,  actual.site_conversion,  "rate"),
            "Phone GCV/Order":  (plan.phone_gcv_order,  actual.phone_gcv_order,  "money"),
            "Cart RR":          (plan.cart_rr,          actual.cart_rr,          "rate"),
            "Cart Conversion":  (plan.cart_conversion,  actual.cart_conversion,  "rate"),
            "Cart GCV/Order":   (plan.cart_gcv_order,   actual.cart_gcv_order,   "money"),
        }
        p, a, kind = mapping.get(name, (None, None, None))
        if p is None:
            return "—", "—", "—"
        if kind == "num":
            return f"{p:,.0f}", f"{a:,.0f}", f"{(a / p - 1) * 100:+.1f}%" if p else "—"
        if kind == "rate":
            return f"{p * 100:.2f}%", f"{a * 100:.2f}%", f"{(a / p - 1) * 100:+.1f}%" if p else "—"
        if kind == "money":
            return f"\\${p:,.2f}", f"\\${a:,.2f}", f"{(a / p - 1) * 100:+.1f}%" if p else "—"
        return "—", "—", "—"

    ranked = sorted(impacts, key=lambda kv: abs(kv[1]), reverse=True)
    ranked_drivers: list[dict] = []
    for i, (name, dollar) in enumerate(ranked, start=1):
        plan_fmt, pacing_fmt, vs_plan = _plan_pacing_formatted(name)
        sign = "-" if dollar < 0 else "+"
        ranked_drivers.append({
            "rank": i,
            "name": name,
            "impact": float(dollar),
            "impact_formatted": f"{sign}\\${abs(dollar):,.0f}",
            "pct_of_gap": f"{dollar / gap * 100:+.0f}%",
            "plan_formatted": plan_fmt,
            "pacing_formatted": pacing_fmt,
            "vs_plan_pct": vs_plan,
        })

    pct_of_plan = (result.total_gap / result.plan_revenue) if result.plan_revenue else 0.0

    return {
        "plan_revenue": float(result.plan_revenue),
        "pacing_revenue": float(result.pacing_revenue),
        "total_gap": float(result.total_gap),
        "pct_of_plan": pct_of_plan,
        "ranked_drivers": ranked_drivers,
    }


# ── Renderer helpers ──────────────────────────────────────────────────────


_FLOWCHART_ASSETS = [
    "assets/seo_flowchart_top.png",
    "assets/seo_flowchart_middle.png",
    "assets/seo_flowchart_bottom.png",
]


def _render_diagnostic_framework(report: DiagnosticReport) -> None:
    """Render the 'Diagnostic framework' expander.

    Two views inside:
      1. Live Graphviz traversal of the flowchart (visited path highlighted).
      2. The manager's original flowchart as static images, for reference.
    """
    with st.expander("Diagnostic framework — flowchart walk", expanded=False):
        st.caption(
            "Highlighted path = gates we traversed given the numbers above. "
            "The solid-orange terminal box is the framework's answer."
        )
        try:
            st.graphviz_chart(report.render_graphviz(), use_container_width=True)
        except Exception as e:  # pragma: no cover — graphviz rendering failure
            st.warning(f"Live flowchart rendering failed: {e}")

        # Show the traversed gate decisions as a compact sequence.
        if report.gate_decisions:
            decision_lines: list[str] = []
            for i, d in enumerate(report.gate_decisions, start=1):
                decision_lines.append(
                    f"**{i}. {d.name}** → `{d.verdict}` · {d.evidence}"
                )
            st.markdown("\n\n".join(decision_lines))

        st.divider()
        st.caption("Reference — manager's original flowchart:")
        for path in _FLOWCHART_ASSETS:
            if os.path.exists(path):
                st.image(path, use_container_width=True)


def _build_session_share_map(
    period: OrganicPeriod, ui_site: str, *, use_bucket: bool = True,
) -> dict[str, float]:
    """Build a {page_type_or_bucket: session_share} map for the current window.

    Used by the flowchart walker (diagnostic evidence) and keyword-mover
    severity scoring to contextualize ranking drops by business volume.
    Falls back to an empty dict on any error so callers degrade gracefully.
    """
    try:
        sess_df = fetch_organic_session_funnel_by_page_type(
            str(period.curr_start), str(period.curr_end),
        )
    except Exception:
        return {}
    if sess_df.empty:
        return {}
    if ui_site != "All":
        sess_df = sess_df[sess_df["site"] == ui_site]
    if sess_df.empty:
        return {}
    col = "landing_page_type_bucket" if (use_bucket and "landing_page_type_bucket" in sess_df.columns) else "landing_page_type"
    pt_sessions = sess_df.groupby(col)["sessions"].sum()
    total = pt_sessions.sum()
    if total <= 0:
        return {}
    return (pt_sessions / total).to_dict()


def _render_mtd_tldr(
    *,
    ctx: AppContext,
    period: OrganicPeriod,
    site_choice: str,
    decomp: dict | None,
    curr_rank: float | None,
    prior_rank: float | None,
    page_type_top_movers: list[dict],
    top_keyword_movers: list[dict],
    organic_sessions: dict | None,
) -> None:
    """MTD TL;DR — lead with revenue waterfall, drill into Sessions only
    when Sessions is a top driver.
    """
    st.header("TL;DR — MTD vs Plan")
    st.caption(
        f"GSC-aligned window: **{period.window_label}** · "
        + (f"GSC through **{period.gsc_max_date}**"
           if period.gsc_max_date else "GSC freshness unknown")
    )
    if period.truncation_note:
        st.caption(f":information_source: {period.truncation_note}")

    waterfall = _build_waterfall_payload(ctx)
    if not waterfall:
        st.warning("Revenue waterfall unavailable — cannot render MTD TL;DR.")
        return

    # Decide whether Sessions is a top driver (if so, run the flowchart).
    top2_drivers = {d["name"] for d in waterfall["ranked_drivers"][:2]}
    sessions_is_driver = "Sessions" in top2_drivers

    page1_churn = _fetch_page1_churn_safe(period, site_choice) if sessions_is_driver else {}
    session_share = _build_session_share_map(period, site_choice)
    # No P4WA for MTD view.
    four_metric_llm, four_metric_df = _build_4metric_table(
        decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
        organic_sessions=organic_sessions, p4wa=None,
    )
    # Strip the P4WA columns for MTD display.
    mtd_table = four_metric_df.drop(columns=["P4WA", "Δ% vs P4WA"])

    report = _run_flowchart(
        period=period, decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
        page_type_top_movers=page_type_top_movers,
        organic_sessions=organic_sessions, page1_churn=page1_churn or {},
        session_share_map=session_share,
    )

    # LLM narrative — THE primary artifact. One paragraph, exec-ready.
    payload = {
        "window": _window_dict(period, site_choice),
        "waterfall": waterfall,
        "diagnostic": report.to_dict() if sessions_is_driver else None,
        "four_metric": four_metric_llm,
        "page_type_movers": page_type_top_movers,
        "top_keyword_movers": top_keyword_movers,
    }
    with st.spinner("Synthesising MTD TL;DR…"):
        try:
            narrative = generate_mtd_vs_plan_tldr(payload)
        except Exception as e:  # pragma: no cover
            narrative = f"*TL;DR unavailable — {e}*"
    st.markdown(narrative)

    # Per-metric detail is secondary — collapsed by default for the exec
    # reader, expanded by analysts who want to verify the paragraph.
    with st.expander("Metric-by-metric detail", expanded=False):
        st.dataframe(mtd_table, use_container_width=True, hide_index=True)

    # Diagnostic framework (only if Sessions is a driver — otherwise the
    # flowchart isn't the right lens for the question).
    if sessions_is_driver:
        _render_diagnostic_framework(report)


def _render_wow_tldr(
    *,
    period: OrganicPeriod,
    site_choice: str,
    decomp: dict | None,
    curr_rank: float | None,
    prior_rank: float | None,
    page_type_top_movers: list[dict],
    top_keyword_movers: list[dict],
    organic_sessions: dict | None,
) -> None:
    """WoW TL;DR — 4-metric table (with P4WA), flowchart walk."""
    st.header("TL;DR — Week over week")
    st.caption(
        f"Current: **{period.curr_start} → {period.curr_end}** "
        f"({period.curr_days}d) · Prior: **{period.prior_start} → {period.prior_end}** · "
        + (f"GSC through **{period.gsc_max_date}**"
           if period.gsc_max_date else "GSC freshness unknown")
    )
    if period.truncation_note:
        st.caption(f":information_source: {period.truncation_note}")

    p4wa = _fetch_p4wa_safe(period, site_choice)
    page1_churn = _fetch_page1_churn_safe(period, site_choice)
    session_share = _build_session_share_map(period, site_choice)
    four_metric_llm, four_metric_df = _build_4metric_table(
        decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
        organic_sessions=organic_sessions, p4wa=p4wa,
    )
    report = _run_flowchart(
        period=period, decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
        page_type_top_movers=page_type_top_movers,
        organic_sessions=organic_sessions, page1_churn=page1_churn,
        session_share_map=session_share,
    )

    # LLM narrative — primary artifact.
    payload = {
        "window": _window_dict(period, site_choice),
        "four_metric": four_metric_llm,
        "diagnostic": report.to_dict(),
        "click_decomp": decomp,
        "page_type_movers": page_type_top_movers,
        "top_keyword_movers": top_keyword_movers,
    }
    with st.spinner("Synthesising WoW TL;DR…"):
        try:
            narrative = generate_wow_tldr(payload)
        except Exception as e:  # pragma: no cover
            narrative = f"*TL;DR unavailable — {e}*"
    st.markdown(narrative)

    # Per-metric detail (secondary).
    with st.expander("Metric-by-metric detail (this week vs prior vs P4WA)",
                      expanded=False):
        st.dataframe(four_metric_df, use_container_width=True, hide_index=True)
        if p4wa and p4wa.get("weeks_used") and p4wa["weeks_used"] < 4:
            st.caption(
                f":information_source: P4WA baseline averaged across "
                f"{p4wa['weeks_used']} weeks (some weeks had no data)."
            )

    _render_diagnostic_framework(report)


def _render_custom_tldr(
    *,
    period: OrganicPeriod,
    site_choice: str,
    decomp: dict | None,
    curr_rank: float | None,
    prior_rank: float | None,
    page_type_top_movers: list[dict],
    top_keyword_movers: list[dict],
    organic_sessions: dict | None,
) -> None:
    """Custom-window TL;DR — WoW format minus P4WA."""
    st.header("TL;DR — Custom window")
    st.caption(
        f"Current: **{period.curr_start} → {period.curr_end}** "
        f"({period.curr_days}d) vs Prior: **{period.prior_start} → {period.prior_end}** "
        f"({period.prior_days}d) · "
        + (f"GSC through **{period.gsc_max_date}**"
           if period.gsc_max_date else "GSC freshness unknown")
    )
    if period.truncation_note:
        st.caption(f":information_source: {period.truncation_note}")

    page1_churn = _fetch_page1_churn_safe(period, site_choice)
    session_share = _build_session_share_map(period, site_choice)
    four_metric_llm, four_metric_df = _build_4metric_table(
        decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
        organic_sessions=organic_sessions, p4wa=None,
    )
    custom_table = four_metric_df.drop(columns=["P4WA", "Δ% vs P4WA"])
    report = _run_flowchart(
        period=period, decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
        page_type_top_movers=page_type_top_movers,
        organic_sessions=organic_sessions, page1_churn=page1_churn,
        session_share_map=session_share,
    )

    payload = {
        "window": _window_dict(period, site_choice),
        "four_metric": four_metric_llm,
        "diagnostic": report.to_dict(),
        "click_decomp": decomp,
        "page_type_movers": page_type_top_movers,
        "top_keyword_movers": top_keyword_movers,
    }
    with st.spinner("Synthesising TL;DR…"):
        try:
            narrative = generate_custom_tldr(payload)
        except Exception as e:  # pragma: no cover
            narrative = f"*TL;DR unavailable — {e}*"
    st.markdown(narrative)

    with st.expander("Metric-by-metric detail", expanded=False):
        st.dataframe(custom_table, use_container_width=True, hide_index=True)

    _render_diagnostic_framework(report)


def _window_dict(period: OrganicPeriod, site_choice: str) -> dict:
    return {
        "mode": period.mode,
        "curr_start": period.curr_start.isoformat(),
        "curr_end": period.curr_end.isoformat(),
        "prior_start": period.prior_start.isoformat(),
        "prior_end": period.prior_end.isoformat(),
        "curr_days": period.curr_days,
        "prior_days": period.prior_days,
        "gsc_max_date": period.gsc_max_date.isoformat() if period.gsc_max_date else None,
        "truncation_note": period.truncation_note,
        "window_label": period.window_label,
        "site_filter": site_choice,
    }


# ---------------------------------------------------------------------------
# Section 3 — Top-of-Funnel Visibility (GSC d_5, four-panel trend)
# ---------------------------------------------------------------------------


def _render_gsc_visibility(
    period: OrganicPeriod, ui_site: str, daily: pd.DataFrame
) -> None:
    st.header("3. Top-of-Funnel Visibility (GSC)")
    trend_start = date(period.curr_start.year - 1, 1, 1)
    _window_badge(
        "15-month trend", trend_start, period.gsc_max_date,
        extra="partial months paced",
    )
    st.caption(
        "Site-level totals from `gsc_search_analytics_d_5` — matches the GSC "
        "dashboard exactly. CTR and weighted rank are impression-weighted "
        "aggregates. Partial months are paced to full-month equivalents."
    )

    if daily is None or daily.empty:
        st.info("No GSC rows returned for the selected site/date range.")
        return

    monthly = aggregate_gsc_daily_to_monthly(daily)
    if monthly.empty:
        st.info("No rollup available — GSC returned zero impressions.")
        return

    # Roll across sites when 'All' so every subplot has ONE combined line.
    if ui_site == "All":
        mo = monthly.copy()
        mo["rank_x_impr"] = mo["weighted_avg_rank"] * mo["impressions"]
        plot_df = (
            mo.groupby("month", as_index=False)
            .agg(
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
                clicks_paced=("clicks_paced", "sum"),
                impressions_paced=("impressions_paced", "sum"),
                rank_x_impr=("rank_x_impr", "sum"),
                days_with_data=("days_with_data", "max"),
                days_in_month=("days_in_month", "max"),
            )
        )
        plot_df["ctr"] = plot_df["clicks"] / plot_df["impressions"].replace(0, pd.NA)
        plot_df["weighted_avg_rank"] = (
            plot_df["rank_x_impr"] / plot_df["impressions"].replace(0, pd.NA)
        )
        plot_df = plot_df.drop(columns=["rank_x_impr"])
        series_label = "All Sites"
    else:
        plot_df = monthly
        series_label = ui_site

    plot_df = plot_df.sort_values("month")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monthly Impressions (paced)",
            "Monthly Clicks (paced)",
            "CTR (%)",
            "Weighted Avg Rank (lower = better)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    color = _PALETTE[0]
    x = pd.to_datetime(plot_df["month"])

    fig.add_trace(
        go.Bar(
            x=x, y=plot_df["impressions_paced"], name=series_label,
            marker_color=color,
            hovertemplate="%{x|%b %Y}<br>Impressions: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=x, y=plot_df["clicks_paced"], name=series_label,
            marker_color=color,
            hovertemplate="%{x|%b %Y}<br>Clicks: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=(plot_df["ctr"] * 100).round(2),
            mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5),
            hovertemplate="%{x|%b %Y}<br>CTR: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=plot_df["weighted_avg_rank"].round(1),
            mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5),
            hovertemplate="%{x|%b %Y}<br>Avg Rank: %{y:.1f}<extra></extra>",
        ),
        row=2, col=2,
    )
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_layout(
        height=620, margin=dict(l=60, r=40, t=70, b=60),
        title_text=f"GSC Visibility — {series_label}",
        title_font=dict(size=17),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(tickformat="%b %Y", dtick="M3", showgrid=False, row=r, col=c)
            fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)", row=r, col=c)

    st.plotly_chart(fig, use_container_width=True)

    latest = plot_df.iloc[-1]
    days_with = int(latest.get("days_with_data", 0) or 0)
    days_in = int(latest.get("days_in_month", 0) or 0)
    partial_note = (
        " *(pacing-projected — partial month)*"
        if days_with and days_in and days_with < days_in
        else ""
    )
    st.caption(
        f"Latest month ({pd.Timestamp(latest['month']).strftime('%b %Y')}): "
        f"**{int(latest['impressions']):,} impressions · "
        f"{int(latest['clicks']):,} clicks · "
        f"{latest['ctr'] * 100:.2f}% CTR · "
        f"avg rank {latest['weighted_avg_rank']:.1f}**"
        + partial_note
    )


# ---------------------------------------------------------------------------
# Section 4 — Click Decomposition
# ---------------------------------------------------------------------------


def _compute_click_decomp_for_window(
    daily: pd.DataFrame, period: OrganicPeriod
) -> tuple[dict | None, float | None, float | None]:
    """Run the click decomposition on the truncated Organic window.

    Returns ``(decomp_dict_or_None, curr_rank, prior_rank)``.
    """
    if daily is None or daily.empty:
        return None, None, None

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    curr = _sum_daily(d, period.curr_start, period.curr_end)
    prior = _sum_daily(d, period.prior_start, period.prior_end)

    if curr["clicks"] == 0 and prior["clicks"] == 0:
        return None, curr.get("weighted_avg_rank"), prior.get("weighted_avg_rank")

    decomp = compute_click_decomposition(
        curr_clicks=curr["clicks"],
        curr_impressions=curr["impressions"],
        prior_clicks=prior["clicks"],
        prior_impressions=prior["impressions"],
    )
    return decomp, curr.get("weighted_avg_rank"), prior.get("weighted_avg_rank")


def _render_click_decomposition(
    period: OrganicPeriod,
    decomp: dict | None,
    curr_rank: float | None,
    prior_rank: float | None,
) -> None:
    st.header("4. Click Decomposition")
    _organic_window_badge(period)
    st.caption(
        "Splits the period-over-period click change into **impression effect** "
        "(Δ impressions × prior CTR) and **CTR effect** (Δ CTR × prior impressions), "
        "then maps it to the diagnostic decision tree."
    )

    if decomp is None:
        st.info("GSC data missing for one of the comparison windows.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Clicks (current)",
        f"{int(decomp['curr_clicks']):,}",
        delta=f"{decomp['delta_clicks']:+,.0f} ({decomp['pct_change_clicks'] * 100:+.1f}%)",
    )
    c2.metric(
        "Impressions (current)",
        f"{int(decomp['curr_impressions']):,}",
        delta=f"{decomp['delta_impressions']:+,.0f} ({decomp['pct_change_impressions'] * 100:+.1f}%)",
    )
    c3.metric(
        "CTR (current)",
        f"{decomp['curr_ctr'] * 100:.2f}%",
        delta=f"{decomp['delta_ctr'] * 100:+.2f}pp ({decomp['pct_change_ctr'] * 100:+.1f}%)",
    )
    c4.metric(
        "Weighted avg rank",
        f"{curr_rank:.1f}" if curr_rank is not None else "—",
        delta=(
            f"{curr_rank - prior_rank:+.1f} positions"
            if (curr_rank is not None and prior_rank is not None) else None
        ),
        delta_color="inverse",  # lower rank = better
    )

    labels = [
        "Prior clicks", "Impression effect", "CTR effect", "Interaction", "Current clicks",
    ]
    values = [
        decomp["prior_clicks"],
        decomp["impression_effect"],
        decomp["ctr_effect"],
        decomp["interaction"],
        decomp["curr_clicks"],
    ]
    measure = ["absolute", "relative", "relative", "relative", "total"]
    fig = go.Figure(
        go.Waterfall(
            x=labels, y=values, measure=measure,
            text=[
                f"<b>{int(decomp['prior_clicks']):,}</b>",
                f"<b>{decomp['impression_effect']:+,.0f}</b>",
                f"<b>{decomp['ctr_effect']:+,.0f}</b>",
                f"<b>{decomp['interaction']:+,.0f}</b>",
                f"<b>{int(decomp['curr_clicks']):,}</b>",
            ],
            textposition="outside",
            connector=dict(line=dict(color="rgb(160,160,160)", width=1, dash="dot")),
            increasing=dict(marker=dict(color="#27ae60")),
            decreasing=dict(marker=dict(color="#e74c3c")),
            totals=dict(marker=dict(color="#2c3e50")),
        )
    )
    fig.update_layout(
        title=f"Clicks · {period.window_label} ({period.curr_days}d each)",
        height=420, margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(tickformat=",.0f", gridcolor="rgba(0,0,0,0.06)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    diagnosis = diagnose_click_change(decomp, curr_rank=curr_rank, prior_rank=prior_rank)
    st.markdown(f"**Diagnosis.** {diagnosis}")

    with st.expander("Decomposition detail"):
        detail = pd.DataFrame(
            [
                {"Metric": "Prior period",
                 "Clicks": f"{int(decomp['prior_clicks']):,}",
                 "Impressions": f"{int(decomp['prior_impressions']):,}",
                 "CTR": f"{decomp['prior_ctr'] * 100:.2f}%",
                 "Avg rank": f"{prior_rank:.1f}" if prior_rank is not None else "—"},
                {"Metric": "Current period",
                 "Clicks": f"{int(decomp['curr_clicks']):,}",
                 "Impressions": f"{int(decomp['curr_impressions']):,}",
                 "CTR": f"{decomp['curr_ctr'] * 100:.2f}%",
                 "Avg rank": f"{curr_rank:.1f}" if curr_rank is not None else "—"},
                {"Metric": "Δ",
                 "Clicks": f"{decomp['delta_clicks']:+,.0f}",
                 "Impressions": f"{decomp['delta_impressions']:+,.0f}",
                 "CTR": f"{decomp['delta_ctr'] * 100:+.2f}pp",
                 "Avg rank": (
                     f"{curr_rank - prior_rank:+.1f}"
                     if (curr_rank is not None and prior_rank is not None) else "—"
                 )},
            ]
        )
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.caption(
            "`impression_effect = Δ impressions × prior CTR`, "
            "`ctr_effect = Δ CTR × prior impressions`, "
            "`interaction = Δ impressions × Δ CTR` (small residual). "
            "The three sum exactly to the click delta."
        )


# ---------------------------------------------------------------------------
# Section 5 — Performance by Landing Page Type (+ Unmatched explainer)
# ---------------------------------------------------------------------------


def _render_page_type_performance(
    period: OrganicPeriod, ui_site: str
) -> list[dict]:
    """Render the page-type breakdown + unmatched-URLs explainer.

    Offers two grouping dimensions:
      • ``landing_page_type_bucket`` (default) — the 6-bucket exec-view
        rollup (Homepage / StateGEO / CityGEO / Provider / PlanType /
        Informational). Matches the taxonomy owned by
        ``bucket_for_landing_page_type`` in ``app/seo_data.py``.
      • ``landing_page_type`` — raw 28-value detail from
        ``mp_session_level_query`` for drill-downs.

    Returns a list of the top 3 movers (absolute click delta) at whichever
    grain is selected so the TL;DR can cite them. Keys are always named
    ``landing_page_type`` in the returned dicts so downstream consumers
    (narrative, diagnostic) keep working regardless of the view mode.
    """
    st.header("5. Performance by Landing Page Type")
    _organic_window_badge(
        period,
        extra=f"Page-type lookup: sessions since {_PAGE_TYPE_LOOKUP_START}",
    )

    # ---------- Bucket-vs-raw toggle ----------
    view_choice = st.radio(
        "View",
        options=["Bucket (6 groups)", "Raw landing_page_type"],
        index=0,
        horizontal=True,
        key="organic_page_type_view",
        help=(
            "Bucket view rolls the 28 `landing_page_type` values in "
            "`mp_session_level_query` into the 6-bucket exec taxonomy "
            "(Homepage, StateGEO, CityGEO, Provider, PlanType, "
            "Informational). Use the raw view for drill-downs."
        ),
    )
    use_bucket = view_choice.startswith("Bucket")
    group_col = "landing_page_type_bucket" if use_bucket else "landing_page_type"

    if use_bucket:
        st.caption(
            "Grouped by `landing_page_type_bucket` (6 exec buckets). "
            "Source: `gsc_search_analytics_d_1` joined to the URL → "
            "`landing_page_type` lookup from `mp_session_level_query`, then "
            "rolled up via `bucket_for_landing_page_type`. 'Unmatched' = "
            "GSC pages that don't appear as an Organic landing URL."
        )
    else:
        st.caption(
            "Raw `landing_page_type` view. Joins "
            "`gsc_search_analytics_d_1` to the URL → `landing_page_type` "
            "lookup built from `mp_session_level_query`. 'Unmatched' = "
            "GSC pages that don't appear as an Organic landing URL."
        )

    domains = _site_domain_filter(ui_site)
    try:
        curr = fetch_gsc_by_page_type(
            str(period.curr_start), str(period.curr_end), domains=domains
        )
        prior = fetch_gsc_by_page_type(
            str(period.prior_start), str(period.prior_end), domains=domains
        )
    except Exception as e:  # pragma: no cover
        st.warning(f"GSC page-type breakdown unavailable: {e}")
        return []

    if curr.empty:
        st.info("No GSC page-type data for the current window.")
        return []

    def _roll(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        tmp = df.copy()
        tmp["pos_x_impr"] = tmp["weighted_avg_rank"] * tmp["impressions"]
        g = (
            tmp.groupby(group_col, as_index=False)
            .agg(
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
                pos_x_impr=("pos_x_impr", "sum"),
            )
        )
        g["ctr"] = g["clicks"] / g["impressions"].replace(0, pd.NA)
        g["weighted_avg_rank"] = g["pos_x_impr"] / g["impressions"].replace(0, pd.NA)
        return g.drop(columns=["pos_x_impr"])

    c_roll = _roll(curr)
    p_roll = _roll(prior)

    merged = c_roll.merge(
        p_roll, on=group_col, how="outer", suffixes=("_curr", "_prior")
    ).fillna(
        {"clicks_curr": 0, "clicks_prior": 0, "impressions_curr": 0, "impressions_prior": 0}
    )
    merged["click_delta"] = merged["clicks_curr"] - merged["clicks_prior"]
    merged["click_delta_pct"] = (
        merged["click_delta"] / merged["clicks_prior"].replace(0, pd.NA)
    )
    merged["impr_delta_pct"] = (
        (merged["impressions_curr"] - merged["impressions_prior"])
        / merged["impressions_prior"].replace(0, pd.NA)
    )

    # Bucket view keeps every bucket even if small (there are only 6 —
    # suppressing noise isn't a concern). Raw view still filters out very
    # small rows so the chart stays readable.
    unmatched_row = merged[merged[group_col] == "Unmatched"]
    if use_bucket:
        material = merged[merged[group_col] != "Unmatched"].copy()
    else:
        material = merged[
            (merged[group_col] != "Unmatched")
            & ((merged["clicks_curr"] >= 20) | (merged["clicks_prior"] >= 20))
        ].copy()
    material = material.sort_values("click_delta")

    if material.empty:
        st.info("No materially-sized classified page types (>= 20 clicks either period).")
    else:
        bar_fig = go.Figure(
            go.Bar(
                y=material[group_col],
                x=material["click_delta"],
                orientation="h",
                marker_color=[
                    "#e74c3c" if v < 0 else "#27ae60" for v in material["click_delta"]
                ],
                text=[
                    f"{v:+,.0f}" + (f" ({p * 100:+.1f}%)" if pd.notna(p) else "")
                    for v, p in zip(material["click_delta"], material["click_delta_pct"])
                ],
                textposition="outside",
            )
        )
        chart_title_suffix = "Bucket" if use_bucket else "Page Type"
        bar_fig.update_layout(
            title=(
                f"GSC Click Change by {chart_title_suffix} · "
                f"{period.window_label}"
            ),
            xaxis_title="Click Δ",
            height=max(360, len(material) * 32 + 80),
            margin=dict(l=180, r=80, t=60, b=50),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        top_loss = material.nsmallest(5, "click_delta")
        top_gain = material.nlargest(5, "click_delta")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 click losses**")
            for _, r in top_loss.iterrows():
                st.markdown(
                    f"- `{r[group_col]}` — **{r['click_delta']:+,.0f}** clicks "
                    f"({r['click_delta_pct'] * 100:+.1f}%)"
                )
        with col2:
            st.markdown("**Top 5 click gains**")
            for _, r in top_gain.iterrows():
                st.markdown(
                    f"- `{r[group_col]}` — **{r['click_delta']:+,.0f}** clicks "
                    f"({r['click_delta_pct'] * 100:+.1f}%)"
                )

        display_header = "Bucket" if use_bucket else "Page Type"
        display = pd.DataFrame(
            {
                display_header: material[group_col],
                "Curr Clicks": material["clicks_curr"].map(lambda v: f"{int(v):,}"),
                "Prior Clicks": material["clicks_prior"].map(lambda v: f"{int(v):,}"),
                "Click Δ": material["click_delta"].map(lambda v: f"{v:+,.0f}"),
                "Click Δ %": material["click_delta_pct"].map(
                    lambda v: f"{v * 100:+.1f}%" if pd.notna(v) else "—"
                ),
                "Curr Impr": material["impressions_curr"].map(lambda v: f"{int(v):,}"),
                "Impr Δ %": material["impr_delta_pct"].map(
                    lambda v: f"{v * 100:+.1f}%" if pd.notna(v) else "—"
                ),
                "Curr CTR": material["ctr_curr"].map(
                    lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "—"
                ),
                "Curr Rank": material["weighted_avg_rank_curr"].map(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                ),
                "Prior Rank": material["weighted_avg_rank_prior"].map(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                ),
            }
        ).sort_values(display_header)
        expander_label = (
            "Bucket detail" if use_bucket else "Full page-type detail"
        )
        with st.expander(expander_label):
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ---------- Unmatched callout ----------
    if not unmatched_row.empty:
        u = unmatched_row.iloc[0]
        total_curr = merged["clicks_curr"].sum()
        share = (u["clicks_curr"] / total_curr) if total_curr else 0.0
        st.warning(
            f"**'Unmatched' accounts for {int(u['clicks_curr']):,} clicks "
            f"({share * 100:.0f}% of total) in the current window.** "
            "Unmatched = GSC pages that don't appear as an Organic landing URL in "
            "`mp_session_level_query` since 2025-01-01."
        )
        with st.expander("Why are pages unmatched?"):
            st.markdown(
                "Typical causes, roughly in order of impact:\n\n"
                "1. **Impression-only pages** — the page ranks in Google but receives "
                "zero or very few organic clicks, so it never appears as a session "
                "landing URL. This is by far the most common cause.\n"
                "2. **Excluded by the session filter** — `session_level_query` filters "
                "out `/resources/`, `/solar-energy/`, non-Texas traffic, bot IPs, and "
                "internal IPs. Any GSC page matching these rules can't map.\n"
                "3. **URL normalisation drift** — we normalise with `RTRIM('/')` + "
                "`LOWER()` + strip `#fragment` on both sides, but query strings "
                "(`?utm_*`, etc.) in GSC `page` values can still slip through.\n"
                "4. **Newly published pages** — anything launched after "
                "`2025-01-01` (the CTE's lookback start) that hasn't yet accrued enough "
                "organic sessions to register.\n\n"
                "Below: the current window's top unmatched URLs by clicks. Spot-check "
                "these against the live site to figure out which bucket each falls into."
            )
            try:
                unmatched_urls = fetch_gsc_unmatched_urls(
                    str(period.curr_start), str(period.curr_end),
                    top_n=20, domains=domains,
                )
            except Exception as e:  # pragma: no cover
                st.warning(f"Unmatched URL list unavailable: {e}")
                unmatched_urls = pd.DataFrame()

            if unmatched_urls.empty:
                st.info("No unmatched URLs returned.")
            else:
                disp = pd.DataFrame({
                    "Site": unmatched_urls["site"],
                    "Page": unmatched_urls["page"],
                    "Clicks": unmatched_urls["clicks"].map(lambda v: f"{int(v):,}"),
                    "Impressions": unmatched_urls["impressions"].map(lambda v: f"{int(v):,}"),
                    "CTR": unmatched_urls["ctr"].map(
                        lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "—"
                    ),
                    "Avg Rank": unmatched_urls["weighted_avg_rank"].map(
                        lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                    ),
                })
                st.dataframe(
                    disp, use_container_width=True, hide_index=True,
                    column_config={
                        "Page": st.column_config.LinkColumn("Page", width="large"),
                    },
                )

    # ---------- Top movers for TL;DR ----------
    # Downstream consumers (narrative + diagnostic) key on
    # `landing_page_type`, so we return that key regardless of which view
    # the user picked. The value is the selected grouping column (bucket
    # name when ``use_bucket`` is True, raw type otherwise).
    if material.empty:
        return []

    # Attach session share so the LLM can contextualize business impact.
    session_share_map: dict[str, float] = {}
    try:
        sess_df = fetch_organic_session_funnel_by_page_type(
            str(period.curr_start), str(period.curr_end),
        )
        if not sess_df.empty:
            if ui_site != "All":
                sess_df = sess_df[sess_df["site"] == ui_site]
            if use_bucket and "landing_page_type_bucket" in sess_df.columns:
                pt_sessions = sess_df.groupby("landing_page_type_bucket")["sessions"].sum()
            else:
                pt_sessions = sess_df.groupby("landing_page_type")["sessions"].sum()
            total_sessions = pt_sessions.sum()
            if total_sessions > 0:
                session_share_map = (pt_sessions / total_sessions).to_dict()
    except Exception:
        pass

    ordered = material.reindex(material["click_delta"].abs().sort_values(ascending=False).index)
    movers: list[dict] = []
    for _, r in ordered.head(3).iterrows():
        lpt = r[group_col]
        sess_share = session_share_map.get(lpt)
        movers.append({
            "landing_page_type": lpt,
            "click_delta": float(r["click_delta"]),
            "click_delta_pct": (
                f"{r['click_delta_pct'] * 100:+.1f}%"
                if pd.notna(r["click_delta_pct"]) else "—"
            ),
            "clicks_curr": float(r["clicks_curr"]),
            "clicks_prior": float(r["clicks_prior"]),
            "session_share_pct": (
                f"{sess_share * 100:.0f}%" if sess_share is not None else None
            ),
        })
    return movers


# ---------------------------------------------------------------------------
# Section 6 — Top Queries by Page Type
# ---------------------------------------------------------------------------


def _render_top_queries(period: OrganicPeriod, ui_site: str) -> None:
    st.header("6. Top Queries by Page Type")
    _organic_window_badge(period)
    st.caption(
        "Top 10 GSC queries by clicks for each `landing_page_type` over the "
        "current window. Source: `gsc_search_analytics_d_1`. Totals here WILL "
        "NOT match the GSC dashboard — Google drops anonymised long-tail queries."
    )

    domains = _site_domain_filter(ui_site)
    try:
        df = fetch_gsc_top_queries_by_page_type(
            str(period.curr_start), str(period.curr_end), top_n=10, domains=domains
        )
    except Exception as e:  # pragma: no cover
        st.warning(f"Top queries unavailable: {e}")
        return

    if df.empty:
        st.info("No query rows returned for the current window.")
        return

    if ui_site == "All":
        df = (
            df.groupby(["landing_page_type", "query"], as_index=False)
            .agg(
                clicks=("clicks", "sum"),
                impressions=("impressions", "sum"),
                avg_rank=("avg_rank", "mean"),
            )
            .sort_values(["landing_page_type", "clicks"], ascending=[True, False])
        )
        df["rn"] = df.groupby("landing_page_type")["clicks"].rank(
            method="first", ascending=False
        )
        df = df[df["rn"] <= 10].drop(columns=["rn"])

    page_types = (
        df.groupby("landing_page_type")["clicks"].sum().sort_values(ascending=False)
    )
    if page_types.empty:
        st.info("No queries for the selected site.")
        return

    pt_pick = st.selectbox(
        "Landing page type",
        options=page_types.index.tolist(),
        index=0,
        help="Sorted by total clicks over the current window.",
    )

    sub = df[df["landing_page_type"] == pt_pick].sort_values("clicks", ascending=False).head(10)
    display = pd.DataFrame({
        "Query": sub["query"],
        "Clicks": sub["clicks"].map(lambda v: f"{int(v):,}"),
        "Impressions": sub["impressions"].map(lambda v: f"{int(v):,}"),
        "Avg Rank": sub["avg_rank"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—"),
    })
    if "site" in sub.columns and ui_site == "All":
        display.insert(0, "Site", sub["site"].values)
    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 6.5 — Top Keyword Ranking Tracker
# ---------------------------------------------------------------------------
#
# Answers "which of our highest-click keywords are seeing real rank
# volatility?" The previous tab reported rank changes at the page-type
# level, which hides the signal — a page-type can look stable in aggregate
# while its top earner drops 4 positions. This section inverts that: top 5
# keywords per page-type by prior-clicks (the ones that mattered going in),
# with side-by-side curr/prior rank + click deltas. Keywords whose rank
# worsened by ≥ 2.0 positions are highlighted in red; improvements by
# ≥ 2.0 are green. The same structured mover list is returned so the
# TL;DR paragraph can name specific keywords and page-types instead of
# stopping at "rankings declined".


# Volatility threshold — absolute Δ rank flagged as a material move. 2.0
# roughly matches a SERP-page shift, which is the scale execs care about.
_KEYWORD_RANK_VOLATILITY = 2.0

# Prior-impressions floor for the tracker. Filters out noisy long-tail
# queries whose rank is meaningless at small sample sizes.
_KEYWORD_MIN_PRIOR_IMPRESSIONS = 500


def _style_rank_delta_cell(delta: float | None) -> str:
    """Inline CSS for a Δ rank cell. Lower rank number = better.

    NaN current rank → keyword dropped out entirely → hard red.
    """
    if delta is None or pd.isna(delta):
        return _EXEC_COLOR_RED
    if abs(delta) < _KEYWORD_RANK_VOLATILITY:
        return _EXEC_COLOR_YELLOW
    return _EXEC_COLOR_GREEN if delta < 0 else _EXEC_COLOR_RED


def _style_click_delta_cell(pct: float | None) -> str:
    """Inline CSS for a Δ clicks % cell. Uses the exec ±2.5% stability band."""
    if pct is None or pd.isna(pct):
        return _EXEC_COLOR_NEUTRAL
    if abs(pct) < _EXEC_PCT_THRESHOLD:
        return _EXEC_COLOR_YELLOW
    return _EXEC_COLOR_GREEN if pct > 0 else _EXEC_COLOR_RED


def _fmt_rank_delta_keyword(delta: float | None, curr_rank: float | None) -> str:
    """Format a rank delta cell. Distinguishes 'dropped out' from 'moved'."""
    if curr_rank is None or pd.isna(curr_rank):
        return "dropped"
    if delta is None or pd.isna(delta):
        return "—"
    return f"{delta:+.1f}"


def _render_top_keyword_tracker(
    period: OrganicPeriod, ui_site: str,
) -> list[dict]:
    """Render the top-5-keyword rank tracker per landing_page_type.

    Returns a ranked list of the biggest keyword movers (|Δ rank| ≥ the
    volatility threshold) for the TL;DR to cite. The returned list is
    sorted by a severity score that combines rank magnitude, prior-period
    click weight, AND the page type's share of Organic sessions — so
    ranking drops on high-traffic page types always outrank drops on
    low-volume templates like Business or Cart.
    """
    st.header("6.5 Top Keyword Ranking Tracker")
    _organic_window_badge(
        period,
        extra=(
            f"Top 5 per page-type by PRIOR clicks · "
            f"≥ {_KEYWORD_MIN_PRIOR_IMPRESSIONS:,} prior impressions · "
            f"red = rank worse by ≥ {_KEYWORD_RANK_VOLATILITY:.0f} positions"
        ),
    )
    st.caption(
        "For each landing-page type, the five queries that drove the most "
        "clicks in the PRIOR window — with current vs prior rank and clicks "
        "side-by-side. Attribution: each query is mapped to the page-type "
        "that received its largest share of prior-window clicks, so a keyword "
        "shows up under the page-type it actually lands on. "
        "Volatility (red/green) cells indicate a move of ≥ "
        f"{_KEYWORD_RANK_VOLATILITY:.0f} positions — roughly a SERP-page shift. "
        "Use this to call out rank losses on the keywords that actually move "
        "clicks, rather than long-tail noise."
    )

    domains = _site_domain_filter(ui_site)
    try:
        df = fetch_gsc_top_keyword_tracker(
            str(period.curr_start), str(period.curr_end),
            str(period.prior_start), str(period.prior_end),
            top_n=5,
            min_prior_impressions=_KEYWORD_MIN_PRIOR_IMPRESSIONS,
            domains=domains,
        )
    except Exception as e:  # pragma: no cover — network/auth degradation
        st.warning(f"Top-keyword ranking tracker unavailable: {e}")
        return []

    if df.empty:
        st.info(
            "No keywords cleared the prior-impressions floor for this window. "
            "Try a wider window or relax the filter."
        )
        return []

    # Aggregate across sites when "All" so each page-type has a single
    # top-5 list. We re-rank by summed prior_clicks per (page_type, query)
    # and recompute impression-weighted rank.
    if ui_site == "All":
        tmp = df.copy()
        for side in ("prior", "curr"):
            tmp[f"{side}_pos_x_impr"] = (
                tmp[f"{side}_rank"].fillna(0) * tmp[f"{side}_impressions"].fillna(0)
            )
        agg = (
            tmp.groupby(["landing_page_type", "query"], as_index=False)
            .agg(
                prior_clicks=("prior_clicks", "sum"),
                prior_impressions=("prior_impressions", "sum"),
                prior_pos_x_impr=("prior_pos_x_impr", "sum"),
                curr_clicks=("curr_clicks", "sum"),
                curr_impressions=("curr_impressions", "sum"),
                curr_pos_x_impr=("curr_pos_x_impr", "sum"),
            )
        )
        agg["prior_rank"] = (
            agg["prior_pos_x_impr"] / agg["prior_impressions"].replace(0, pd.NA)
        )
        agg["curr_rank"] = (
            agg["curr_pos_x_impr"] / agg["curr_impressions"].replace(0, pd.NA)
        )
        agg["rank_delta"] = agg["curr_rank"] - agg["prior_rank"]
        agg["click_delta"] = agg["curr_clicks"] - agg["prior_clicks"]
        agg["click_delta_pct"] = agg.apply(
            lambda r: (r["click_delta"] / r["prior_clicks"])
            if r["prior_clicks"] else None,
            axis=1,
        )
        agg["prior_rank_rank"] = (
            agg.groupby("landing_page_type")["prior_clicks"]
            .rank(method="first", ascending=False)
        )
        agg = agg[agg["prior_rank_rank"] <= 5].copy()
        df = agg.drop(columns=["prior_pos_x_impr", "curr_pos_x_impr"])
        df["site"] = "All"

    # Sort page-types by TOTAL prior-window clicks of the top-5 keywords
    # so the reader sees the biggest-traffic templates first.
    pt_order = (
        df.groupby("landing_page_type")["prior_clicks"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not pt_order:
        st.info("No page-types matched the filter for the current window.")
        return []

    # ---------- One expander per page-type ----------
    for pt in pt_order:
        sub = df[df["landing_page_type"] == pt].sort_values(
            "prior_rank_rank", ascending=True
        )
        red_count = int(
            sub.apply(
                lambda r: (
                    pd.isna(r["curr_rank"])
                    or (pd.notna(r["rank_delta"])
                        and r["rank_delta"] >= _KEYWORD_RANK_VOLATILITY)
                ), axis=1,
            ).sum()
        )
        green_count = int(
            sub.apply(
                lambda r: (
                    pd.notna(r["rank_delta"])
                    and r["rank_delta"] <= -_KEYWORD_RANK_VOLATILITY
                ), axis=1,
            ).sum()
        )
        badge_bits: list[str] = []
        if red_count:
            badge_bits.append(f"{red_count} ↓")
        if green_count:
            badge_bits.append(f"{green_count} ↑")
        badge = f"  ·  {' · '.join(badge_bits)}" if badge_bits else ""
        label = f"**{pt}**  —  {int(sub['prior_clicks'].sum()):,} prior clicks{badge}"

        # Auto-expand when any keyword moved materially.
        with st.expander(label, expanded=bool(badge_bits)):
            header = [
                "Keyword", "Prior Clicks", "Curr Clicks", "Δ Clicks",
                "Prior Rank", "Curr Rank", "Δ Rank",
            ]
            rows_html: list[str] = []
            for _, r in sub.iterrows():
                rank_cell_style = _style_rank_delta_cell(r["rank_delta"])
                click_cell_style = _style_click_delta_cell(r["click_delta_pct"])
                cells: list[tuple[str, str]] = [
                    (str(r["query"]), ""),
                    (_fmt_int_or_dash(r["prior_clicks"]), ""),
                    (_fmt_int_or_dash(r["curr_clicks"]), ""),
                    (
                        _fmt_pct_change(r["click_delta_pct"]),
                        click_cell_style,
                    ),
                    (_fmt_rank_value(r["prior_rank"]), ""),
                    (_fmt_rank_value(r["curr_rank"]), ""),
                    (
                        _fmt_rank_delta_keyword(r["rank_delta"], r["curr_rank"]),
                        rank_cell_style,
                    ),
                ]
                rows_html.append(_exec_table_row(cells))

            html = (
                f'<table style="{_EXEC_TABLE_STYLE}">'
                f"{_exec_table_header(header)}"
                f'<tbody>{"".join(rows_html)}</tbody>'
                "</table>"
            )
            st.markdown(html, unsafe_allow_html=True)

    # ---------- Movers payload for the TL;DR ----------
    #
    # Severity = |Δ rank| × log(prior_clicks + 1) × session_weight.
    # The session_weight is the page type's share of total Organic
    # sessions (floored at 0.05 so tiny page types don't dominate just
    # because of a big rank move). This ensures we never lead the exec
    # summary with a rank drop on a 2%-of-sessions page type like
    # Business when Homepage or CityGEO are also moving.
    movers_df = df.copy()
    movers_df["_effective_delta"] = movers_df["rank_delta"].fillna(
        _KEYWORD_RANK_VOLATILITY * 5
    )
    movers_df = movers_df[
        movers_df["_effective_delta"].abs() >= _KEYWORD_RANK_VOLATILITY
    ].copy()
    if movers_df.empty:
        return []

    # Fetch Organic session counts by page type for session-volume weighting.
    session_weight_map: dict[str, float] = {}
    try:
        sess_df = fetch_organic_session_funnel_by_page_type(
            str(period.curr_start), str(period.curr_end),
        )
        if not sess_df.empty:
            if ui_site != "All":
                sess_df = sess_df[sess_df["site"] == ui_site]
            pt_sessions = sess_df.groupby("landing_page_type")["sessions"].sum()
            total_sessions = pt_sessions.sum()
            if total_sessions > 0:
                session_weight_map = (pt_sessions / total_sessions).to_dict()
    except Exception:
        pass

    def _session_weight(landing_page_type: str) -> float:
        """Session share for a page type, floored at 0.05."""
        return max(session_weight_map.get(landing_page_type, 0.05), 0.05)

    movers_df["severity"] = movers_df.apply(
        lambda r: abs(r["_effective_delta"])
        * math.log(max(float(r["prior_clicks"]), 0.0) + 1.0)
        * _session_weight(str(r["landing_page_type"])),
        axis=1,
    )
    movers_df = movers_df.sort_values("severity", ascending=False)

    movers: list[dict] = []
    for _, r in movers_df.head(5).iterrows():
        curr_rank = (
            float(r["curr_rank"]) if pd.notna(r["curr_rank"]) else None
        )
        rank_delta = (
            float(r["rank_delta"]) if pd.notna(r["rank_delta"]) else None
        )
        lpt = str(r["landing_page_type"])
        sess_share = session_weight_map.get(lpt)
        movers.append({
            "query": str(r["query"]),
            "landing_page_type": lpt,
            "prior_rank": float(r["prior_rank"]) if pd.notna(r["prior_rank"]) else None,
            "curr_rank": curr_rank,
            "rank_delta": rank_delta,
            "rank_delta_formatted": _fmt_rank_delta_keyword(
                rank_delta, curr_rank,
            ),
            "prior_clicks": float(r["prior_clicks"]),
            "curr_clicks": float(r["curr_clicks"]),
            "click_delta": float(r["click_delta"]),
            "click_delta_pct": (
                f"{r['click_delta_pct'] * 100:+.1f}%"
                if pd.notna(r["click_delta_pct"]) else "—"
            ),
            "dropped_out": curr_rank is None,
            "session_share_pct": (
                f"{sess_share * 100:.0f}%" if sess_share is not None else None
            ),
        })
    return movers


# ---------------------------------------------------------------------------
# Section 7 — Session Funnel by Page Type
# ---------------------------------------------------------------------------


def _render_session_funnel(period: OrganicPeriod, ui_site: str) -> dict | None:
    """Render the Organic session funnel by page type; return headline totals.

    Same Bucket / Raw toggle as Section 5 so the funnel can be read at the
    6-bucket exec grain or drilled down to the 28-value detail.
    """
    st.header("7. Session Funnel by Page Type")
    _organic_window_badge(period, current_only=True)

    view_choice = st.radio(
        "View",
        options=["Bucket (6 groups)", "Raw landing_page_type"],
        index=0,
        horizontal=True,
        key="organic_session_funnel_view",
        help=(
            "Bucket view groups raw `landing_page_type` values via "
            "`bucket_for_landing_page_type` (Homepage / StateGEO / CityGEO / "
            "Provider / PlanType / Informational). Raw view keeps the "
            "28-value detail."
        ),
    )
    use_bucket = view_choice.startswith("Bucket")
    group_col = "landing_page_type_bucket" if use_bucket else "landing_page_type"

    if use_bucket:
        st.caption(
            "Organic funnel from `mp_session_level_query` over the current "
            "window, grouped by the 6-bucket taxonomy. Sessions → ZIPs → "
            "carts → orders and the derived rates (ZLUR, Cart RR, VC)."
        )
    else:
        st.caption(
            "Organic funnel from `mp_session_level_query` over the current "
            "window. Shows sessions → ZIPs → carts → orders and the derived "
            "rates (ZLUR, Cart RR, VC) per `landing_page_type`. Filtered to "
            "≥10 sessions."
        )

    try:
        df = fetch_organic_session_funnel_by_page_type(
            str(period.curr_start), str(period.curr_end)
        )
    except Exception as e:  # pragma: no cover
        st.warning(f"Session funnel unavailable: {e}")
        return None

    if df.empty:
        st.info("No organic sessions in the current window.")
        return None

    if ui_site != "All":
        df = df[df["site"] == ui_site]
    if df.empty:
        st.info(f"No organic sessions for {ui_site} in the current window.")
        return None

    roll = (
        df.groupby(group_col, as_index=False).agg(
            sessions=("sessions", "sum"),
            zip_entries=("zip_entries", "sum"),
            carts=("carts", "sum"),
            orders=("orders", "sum"),
            cart_orders=("cart_orders", "sum"),
            phone_orders=("phone_orders", "sum"),
        )
    )

    roll["zlur_pct"] = roll["zip_entries"] / roll["sessions"].replace(0, pd.NA)
    roll["cart_rate_pct"] = roll["carts"] / roll["sessions"].replace(0, pd.NA)
    roll["vc_pct"] = roll["orders"] / roll["sessions"].replace(0, pd.NA)
    roll = roll.sort_values("sessions", ascending=False)

    total_sessions = int(roll["sessions"].sum())
    total_orders = int(roll["orders"].sum())
    all_vc = (total_orders / total_sessions) if total_sessions else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Organic sessions", f"{total_sessions:,}")
    c2.metric("Organic orders", f"{total_orders:,}")
    c3.metric("Organic VC", f"{all_vc * 100:.2f}%")

    # In bucket mode there are only ~7 rows — show them all; in raw mode
    # trim to the top 10 so the chart stays readable.
    chart_rows = roll if use_bucket else roll.head(10)
    chart_rows = chart_rows.sort_values("sessions")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Organic Sessions", "VC %"),
        horizontal_spacing=0.15,
    )
    fig.add_trace(
        go.Bar(
            y=chart_rows[group_col], x=chart_rows["sessions"],
            orientation="h", marker_color=_PALETTE[0],
            text=[f"{int(v):,}" for v in chart_rows["sessions"]],
            textposition="outside",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            y=chart_rows[group_col], x=(chart_rows["vc_pct"] * 100).round(2),
            orientation="h", marker_color=_PALETTE[2],
            text=[f"{v * 100:.2f}%" for v in chart_rows["vc_pct"]],
            textposition="outside",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        height=max(400, len(chart_rows) * 36 + 80),
        margin=dict(l=140, r=60, t=60, b=40),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    header_label = "Bucket" if use_bucket else "Page Type"
    display = pd.DataFrame({
        header_label: roll[group_col],
        "Sessions": roll["sessions"].map(lambda v: f"{int(v):,}"),
        "ZIPs": roll["zip_entries"].map(lambda v: f"{int(v):,}"),
        "Carts": roll["carts"].map(lambda v: f"{int(v):,}"),
        "Orders": roll["orders"].map(lambda v: f"{int(v):,}"),
        "ZLUR": roll["zlur_pct"].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "—"),
        "Cart RR": roll["cart_rate_pct"].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "—"),
        "VC": roll["vc_pct"].map(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "—"),
    })
    expander_label = (
        "Full bucket detail" if use_bucket else "Full session funnel detail"
    )
    with st.expander(expander_label):
        st.dataframe(display, use_container_width=True, hide_index=True)

    return {"sessions": total_sessions, "orders": total_orders, "vc_pct": all_vc}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(ctx: AppContext) -> None:
    st.title("Deep Dive — Organic (SEO)")
    _render_sidebar_note()

    # ── Tab-local window picker (drives Sections 3–7) + shared site filter ──
    st.subheader("Organic comparison window")
    mode, curr_range, prior_range = _render_period_picker()
    period = _build_organic_period(mode, curr_range, prior_range)

    # Persistent window banner — lists every time system used on the tab so
    # the reader can always tell which window a given section is reporting on.
    _render_window_banner(ctx, period)

    site_choice = st.radio(
        "Site filter (Sections 0, 3–7)",
        options=GSC_SITE_OPTIONS,
        index=0,
        horizontal=True,
        key="organic_gsc_site",
        help=(
            "Filters the Executive Overview, GSC sections, and session "
            "sections to a single brand. Leave on **All** for the combined "
            "Texas portfolio. Finance (Sections 1–2) is always sidebar-driven."
        ),
    )

    # ── Placeholder for the TL;DR — rendered AFTER we compute everything it needs ──
    tldr_slot = st.container()

    # ── Executive Performance Overview (Section 0) — respects the Site
    #     filter above. We need organic_sessions here too (for the TL;DR),
    #     so compute it up front. ──
    organic_sessions = _compute_organic_sessions_delta(ctx, period)
    st.divider()
    _render_exec_overview(period, organic_sessions, site_choice)

    # ── Finance (Sections 1–2) is sidebar-driven. Render it first so the
    #     page has the full pacing/waterfall even if GSC is slow. ──
    st.divider()
    _render_pacing_summary(ctx)
    st.divider()
    _render_waterfall(ctx)
    st.divider()

    # ── GSC + session sections use `period`. Pull the trend frame once,
    #     reuse across Sections 3 + 4 so the decomposition is guaranteed
    #     consistent with the trend chart. Each section is individually
    #     guarded so partial data doesn't take down the whole tab. ──
    daily = None
    decomp = curr_rank = prior_rank = None
    top_movers = None
    top_keyword_movers = None

    try:
        daily = _load_gsc_daily_trends(period, site_choice)
        _render_gsc_visibility(period, site_choice, daily)
    except Exception as e:
        st.warning(f"GSC visibility data unavailable: {e}")
    st.divider()

    try:
        if daily is not None:
            decomp, curr_rank, prior_rank = _compute_click_decomp_for_window(daily, period)
            _render_click_decomposition(period, decomp, curr_rank, prior_rank)
        else:
            st.info("Click decomposition skipped — GSC data not loaded.")
    except Exception as e:
        st.warning(f"Click decomposition unavailable: {e}")
    st.divider()

    try:
        top_movers = _render_page_type_performance(period, site_choice)
    except Exception as e:
        st.warning(f"Page type performance unavailable: {e}")
    st.divider()

    try:
        _render_top_queries(period, site_choice)
    except Exception as e:
        st.warning(f"Top queries unavailable: {e}")
    st.divider()

    try:
        top_keyword_movers = _render_top_keyword_tracker(period, site_choice)
    except Exception as e:
        st.warning(f"Keyword tracker unavailable: {e}")
    st.divider()

    try:
        _render_session_funnel(period, site_choice)
    except Exception as e:
        st.warning(f"Session funnel unavailable: {e}")

    # ── Now dispatch the mode-specific TL;DR at the top slot
    #     (`organic_sessions` was computed above for the overview). ──
    with tldr_slot:
        if period.mode == "MoM MTD":
            _render_mtd_tldr(
                ctx=ctx, period=period, site_choice=site_choice,
                decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
                page_type_top_movers=top_movers,
                top_keyword_movers=top_keyword_movers,
                organic_sessions=organic_sessions,
            )
        elif period.mode == "WoW":
            _render_wow_tldr(
                period=period, site_choice=site_choice,
                decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
                page_type_top_movers=top_movers,
                top_keyword_movers=top_keyword_movers,
                organic_sessions=organic_sessions,
            )
        else:  # "Custom"
            _render_custom_tldr(
                period=period, site_choice=site_choice,
                decomp=decomp, curr_rank=curr_rank, prior_rank=prior_rank,
                page_type_top_movers=top_movers,
                top_keyword_movers=top_keyword_movers,
                organic_sessions=organic_sessions,
            )
