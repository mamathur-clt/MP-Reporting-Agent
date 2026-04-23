"""
Streamlit app: Energy Marketplace Performance Reporting (v2)

Main entry point. Builds the sidebar controls, loads session + finance data
once, then dispatches to the four v2 tabs:

  1. Overview                — all-channels KPI driver analysis (v1 Sections 1–7)
  2. Deep Dive — Organic     — stubbed; Phase 3 fills in GSC + page-type analysis
  3. Deep Dive — Paid Search — stubbed; Phase 4 fills in campaign-bucket analysis
  4. Ask the Analyst         — chat with tool calling (pandas + Databricks SQL)

Each tab module exposes `render(ctx: AppContext)` and receives a shared
`AppContext` with sidebar state + loaded DataFrames. See `app/app_context.py`.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from app.app_context import AppContext
from app.config import DEFAULT_CHANNELS, KPIS, TIME_MODES
from app.data import fetch_session_data
from app.decomposition import _initiative_label
from app.finance_data import fetch_finance_daily, fetch_plan_pacing
from app.tabs import analyst_chat, organic_deep_dive, overview, paid_search_deep_dive
from app.time_periods import resolve_periods

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy MP – Performance Reporting",
    page_icon="⚡",
    layout="wide",
)

st.title("Energy Marketplace — Performance Reporting")
st.caption(
    "Pacing, period-over-period KPI drivers, and channel-specific deep dives. "
    "Select a KPI, channel(s), and time window in the sidebar, then explore the tabs."
)

# ---------------------------------------------------------------------------
# Sidebar: core controls (KPI + time mode). Filters are rendered post-load
# once we know which channels/websites exist in the data.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    kpi_key = st.selectbox(
        "KPI to analyze",
        list(KPIS.keys()),
        index=list(KPIS.keys()).index("VC"),
        help=KPIS["VC"].description,
    )

    time_mode = st.selectbox("Time comparison", list(TIME_MODES.keys()))
    st.caption(TIME_MODES[time_mode])

    custom_current = custom_prior = None
    if time_mode == "Custom":
        st.subheader("Current period")
        c_start = st.date_input("Current start", value=date.today() - timedelta(days=7))
        c_end = st.date_input("Current end", value=date.today() - timedelta(days=1))
        custom_current = (c_start, c_end)
        st.subheader("Prior period")
        p_start = st.date_input("Prior start", value=date.today() - timedelta(days=14))
        p_end = st.date_input("Prior end", value=date.today() - timedelta(days=8))
        custom_prior = (p_start, p_end)

    st.divider()

    try:
        curr_start, curr_end, prior_start, prior_end = resolve_periods(
            time_mode, custom_current=custom_current, custom_prior=custom_prior
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.metric("Current period", f"{curr_start} → {curr_end}")
    st.metric("Prior period", f"{prior_start} → {prior_end}")

# ---------------------------------------------------------------------------
# Load session-level data (Tier 2). The `step_completions` CTE uses a strict
# `>` comparison on start_date, so we pad the query window back by a day.
# ---------------------------------------------------------------------------
query_start = min(curr_start, prior_start) - timedelta(days=1)
query_end = max(curr_end, prior_end)

with st.spinner("Loading session data from Databricks…"):
    df_all = fetch_session_data(str(query_start), str(query_end))

if df_all.empty:
    st.error("No data returned from Databricks. Check your date range and credentials.")
    st.stop()

available_channels = sorted(df_all["marketing_channel"].dropna().unique().tolist())
available_websites = sorted(df_all["website"].dropna().unique().tolist())
default_ch = [c for c in DEFAULT_CHANNELS if c in available_channels]

# ---------------------------------------------------------------------------
# Sidebar: data-dependent filters (rendered now that we know the universe).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Filters")
    channel_filter = st.multiselect(
        "Marketing channel",
        options=available_channels,
        default=default_ch,
        help="Defaults to Paid Search, Direct, Organic, pMax. Overridden on deep-dive tabs.",
    )
    website_filter = st.multiselect(
        "Website",
        options=available_websites,
        default=[],
        help="Leave empty for all websites.",
    )

# ---------------------------------------------------------------------------
# Filter + split into current / prior frames used by every tab.
# ---------------------------------------------------------------------------
df_filtered = df_all.copy()
if channel_filter:
    df_filtered = df_filtered[df_filtered["marketing_channel"].isin(channel_filter)]
if website_filter:
    df_filtered = df_filtered[df_filtered["website"].isin(website_filter)]

df_current = df_filtered[
    (df_filtered["session_start_date_est"] >= curr_start)
    & (df_filtered["session_start_date_est"] <= curr_end)
].copy()
df_prior = df_filtered[
    (df_filtered["session_start_date_est"] >= prior_start)
    & (df_filtered["session_start_date_est"] <= prior_end)
].copy()

if df_current.empty or df_prior.empty:
    st.warning(
        "One of the periods has no data after filtering. "
        "Try a different time mode or broaden your filters."
    )
    st.stop()

df_current["_initiative_label"] = df_current.apply(_initiative_label, axis=1)
df_prior["_initiative_label"] = df_prior.apply(_initiative_label, axis=1)

# ---------------------------------------------------------------------------
# Load finance data (Tier 1). Non-fatal on failure — the deep-dive and
# overview tabs degrade gracefully when this is missing.
# ---------------------------------------------------------------------------
finance_df = None
plan_df = None
try:
    with st.spinner("Loading finance performance data…"):
        finance_df = fetch_finance_daily()
        plan_df = fetch_plan_pacing()
except Exception as e:  # pragma: no cover - surfaces in UI
    st.warning(f"Finance data unavailable: {e}")

# ---------------------------------------------------------------------------
# Assemble shared context and route to tabs.
# ---------------------------------------------------------------------------
ctx = AppContext(
    kpi_key=kpi_key,
    time_mode=time_mode,
    channel_filter=channel_filter,
    website_filter=website_filter,
    available_channels=available_channels,
    available_websites=available_websites,
    curr_start=curr_start,
    curr_end=curr_end,
    prior_start=prior_start,
    prior_end=prior_end,
    query_start=query_start,
    query_end=query_end,
    df_all=df_all,
    df_filtered=df_filtered,
    df_current=df_current,
    df_prior=df_prior,
    finance_df=finance_df,
    plan_df=plan_df,
)

tab_overview, tab_organic, tab_paid, tab_chat = st.tabs([
    "Overview",
    "Deep Dive — Organic",
    "Deep Dive — Paid Search",
    "Ask the Analyst",
])

with tab_overview:
    overview.render(ctx)

with tab_organic:
    organic_deep_dive.render(ctx)

with tab_paid:
    paid_search_deep_dive.render(ctx)

with tab_chat:
    analyst_chat.render(ctx)

# ---------------------------------------------------------------------------
# Footer — data range + reconciliation check
# ---------------------------------------------------------------------------
st.divider()

footer_cols = st.columns([3, 2])

with footer_cols[0]:
    st.caption(
        f"Data range queried: {query_start} to {query_end} · "
        f"Total sessions loaded: {len(df_all):,} · "
        f"Current period sessions (session-level): {len(df_current):,} · "
        f"Prior period sessions (session-level): {len(df_prior):,}"
    )

with footer_cols[1]:
    if finance_df is not None and not finance_df.empty:
        try:
            fin_curr = finance_df[
                (finance_df["TheDate"] >= str(curr_start))
                & (finance_df["TheDate"] <= str(curr_end))
            ]
            if channel_filter:
                fin_curr = fin_curr[fin_curr["marketing_channel"].isin(channel_filter)]
            fin_sessions = int(fin_curr["sessions"].sum()) if "sessions" in fin_curr.columns else None
            sl_sessions = len(df_current)

            if fin_sessions is not None and sl_sessions > 0:
                delta_pct = (sl_sessions - fin_sessions) / fin_sessions * 100 if fin_sessions else 0
                color = "green" if abs(delta_pct) < 1 else ("orange" if abs(delta_pct) < 3 else "red")
                st.caption(
                    f"Reconciliation check — Current period sessions: "
                    f"Finance={fin_sessions:,}, Session-level={sl_sessions:,} "
                    f"(:{color}[{delta_pct:+.1f}%])"
                )
            else:
                st.caption("Reconciliation: Finance session data not available for comparison.")
        except Exception:
            st.caption("Reconciliation: Could not compare finance vs session-level sessions.")
    else:
        st.caption(
            "Reconciliation: Finance data unavailable — session-level data only."
        )
