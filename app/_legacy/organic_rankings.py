"""
Streamlit page: SEO Organic Ranking Dashboard

Visual, high-level view of ranking performance by domain and top pages.
Key views:
  1. Weighted average rank over time (line chart)
  2. Position distribution over time (stacked area)
  3. Page scorecard (table)
  4. Keyword-level detail (expandable)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

from app.seo_data import (
    fetch_seo_rankings,
    agg_weighted_rank_over_time,
    agg_position_distribution,
    agg_page_scorecard,
    weighted_avg_rank,
    DEFAULT_DOMAINS,
    POSITION_BUCKETS,
    default_seo_start_date,
)

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SEO Organic Rankings",
    page_icon="📈",
    layout="wide",
)

st.title("SEO Organic Ranking Dashboard")
st.caption(
    "Weighted-average rank trends, position distributions, and page-level "
    "scorecards — powered by Clarity keyword tracking data."
)

# ── Sidebar: filters ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    start_date = st.date_input(
        "Data start date",
        value=default_seo_start_date(),
        help="How far back to pull ranking data. Wider windows = slower query.",
    )

    device = st.radio("Device", ["mobile", "desktop"], index=0, horizontal=True)

    domains = st.multiselect(
        "Domains",
        options=DEFAULT_DOMAINS,
        default=DEFAULT_DOMAINS,
        help="Which domains to track rankings for.",
    )

    freq = st.radio(
        "Time granularity",
        ["Weekly", "Daily"],
        index=0,
        horizontal=True,
        help="Aggregate data weekly (less noise) or daily.",
    )
    freq_code = "W" if freq == "Weekly" else "D"

    group_by = st.radio(
        "Group charts by",
        ["Domain", "Page"],
        index=0,
        horizontal=True,
        help="Show one line per domain or per individual page.",
    )
    group_col = "domain" if group_by == "Domain" else "page_label"

if not domains:
    st.warning("Select at least one domain in the sidebar.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────

with st.spinner("Loading SEO ranking data from Databricks…"):
    df = fetch_seo_rankings(
        start_date=str(start_date),
        domains=domains,
        device=device,
    )

if df.empty:
    st.error("No ranking data returned. Check your date range and domain selection.")
    st.stop()

# Populate page filter after data loads
available_pages = sorted(df["page_label"].dropna().unique().tolist())
with st.sidebar:
    st.divider()
    page_filter = st.multiselect(
        "Pages",
        options=available_pages,
        default=[],
        help="Optionally narrow to specific pages. Empty = all pages.",
    )

    available_tags = sorted(
        {
            tag.strip()
            for tags in df["keyword_tags"].dropna().unique()
            for tag in str(tags).split(",")
            if tag.strip()
        }
    )
    tag_filter = st.multiselect(
        "Keyword tag groups",
        options=available_tags,
        default=[],
        help="Filter to keywords matching specific tag groups.",
    )

# Apply post-load filters
if page_filter:
    df = df[df["page_label"].isin(page_filter)]
if tag_filter:
    mask = df["keyword_tags"].fillna("").apply(
        lambda t: any(tag in t for tag in tag_filter)
    )
    df = df[mask]

if df.empty:
    st.warning("No data after applying filters. Broaden your selection.")
    st.stop()

# ── Headline metrics ──────────────────────────────────────────────────────

max_date = max(df["date"])
recent = df[df["date"] == max_date]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest date", str(max_date))
col2.metric("Keywords tracked", f"{df['keyword_tracked'].nunique():,}")
col3.metric(
    "Weighted avg rank (latest)",
    f"{weighted_avg_rank(recent):.1f}" if weighted_avg_rank(recent) else "N/A",
)
col4.metric(
    "Keywords in top 10 (latest)",
    f"{recent[recent['organic_results_web_rank'] <= 10]['keyword_tracked'].nunique():,}",
)

st.divider()

# ── Section 1: Weighted Average Rank Over Time ────────────────────────────

st.header("1. Weighted Average Rank Over Time")
st.caption(
    "Rank weighted by estimated monthly search volume — a keyword with "
    "10K searches matters more than one with 10. Lower is better."
)

ts = agg_weighted_rank_over_time(df, group_col=group_col, freq=freq_code)

fig_rank = go.Figure()
for name, grp in ts.groupby(group_col):
    fig_rank.add_trace(go.Scatter(
        x=grp["period"],
        y=grp["weighted_avg_rank"],
        mode="lines+markers",
        name=str(name),
        hovertemplate=(
            "%{x}<br>"
            f"{group_by}: {name}<br>"
            "Weighted Avg Rank: %{y:.1f}<br>"
            "<extra></extra>"
        ),
    ))

fig_rank.update_layout(
    yaxis=dict(
        title="Weighted Average Rank",
        autorange="reversed",
        gridcolor="rgba(200,200,200,0.3)",
    ),
    xaxis=dict(title=""),
    height=480,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_rank, use_container_width=True)

st.divider()

# ── Section 2: Position Distribution Over Time ───────────────────────────

st.header("2. Keyword Position Distribution Over Time")
st.caption(
    "How many keywords fall into each rank bucket over time. "
    "Growing counts in positions 1-5 signals improving visibility."
)

BUCKET_COLORS = {
    "1": "#1a9641",
    "2": "#66bd63",
    "3-5": "#a6d96a",
    "6-10": "#fee08b",
    "11-20": "#fdae61",
    "20+": "#d73027",
}

dist = agg_position_distribution(df, group_col=group_col, freq=freq_code)
groups_for_dist = sorted(dist[group_col].unique())

if len(groups_for_dist) <= 4:
    cols = st.columns(min(len(groups_for_dist), 2))
    for idx, grp_name in enumerate(groups_for_dist):
        with cols[idx % len(cols)]:
            st.subheader(str(grp_name))
            sub = dist[dist[group_col] == grp_name]
            fig_dist = go.Figure()
            for bucket in POSITION_BUCKETS:
                bdata = sub[sub["position_bucket"] == bucket]
                fig_dist.add_trace(go.Bar(
                    x=bdata["period"],
                    y=bdata["keyword_count"],
                    name=f"Pos {bucket}",
                    marker_color=BUCKET_COLORS[bucket],
                    hovertemplate=f"Pos {bucket}: " + "%{y}<extra></extra>",
                ))
            fig_dist.update_layout(
                barmode="stack",
                yaxis_title="Keyword count",
                xaxis_title="",
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
else:
    dist_selector = st.selectbox(
        "Select group for distribution chart",
        groups_for_dist,
    )
    sub = dist[dist[group_col] == dist_selector]
    fig_dist = go.Figure()
    for bucket in POSITION_BUCKETS:
        bdata = sub[sub["position_bucket"] == bucket]
        fig_dist.add_trace(go.Bar(
            x=bdata["period"],
            y=bdata["keyword_count"],
            name=f"Pos {bucket}",
            marker_color=BUCKET_COLORS[bucket],
            hovertemplate=f"Pos {bucket}: " + "%{y}<extra></extra>",
        ))
    fig_dist.update_layout(
        barmode="stack",
        yaxis_title="Keyword count",
        xaxis_title="",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ── Section 3: Page Scorecard ─────────────────────────────────────────────

st.header("3. Page Scorecard")
st.caption(
    "Current vs prior 7-day weighted average rank for each tracked page. "
    "Negative rank change = improvement (moved up)."
)

scorecard = agg_page_scorecard(df, latest_n_days=7, prior_n_days=7)

if not scorecard.empty:
    display_sc = scorecard.copy()
    display_sc = display_sc.rename(columns={
        "page_label": "Page",
        "domain": "Domain",
        "weighted_avg_rank": "Weighted Avg Rank",
        "prior_weighted_avg_rank": "Prior Wt Avg Rank",
        "rank_change": "Rank Change",
        "avg_web_rank": "Avg Web Rank",
        "keywords_tracked": "Keywords",
        "keywords_in_top_10": "In Top 10",
        "total_search_volume": "Search Volume",
    })

    def _color_rank_change(val):
        if pd.isna(val):
            return ""
        if val < 0:
            return "color: #1a9641; font-weight: bold"
        if val > 0:
            return "color: #d73027; font-weight: bold"
        return ""

    styled = display_sc.style.map(
        _color_rank_change, subset=["Rank Change"]
    ).format({
        "Weighted Avg Rank": "{:.1f}",
        "Prior Wt Avg Rank": "{:.1f}",
        "Rank Change": "{:+.1f}",
        "Avg Web Rank": "{:.1f}",
        "Search Volume": "{:,.0f}",
    }, na_rep="–")

    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("Not enough data for a scorecard comparison.")

st.divider()

# ── Section 4: Keyword-Level Detail ──────────────────────────────────────

st.header("4. Keyword-Level Detail")

with st.expander("Show keyword-level data", expanded=False):
    kw_page_sel = st.selectbox(
        "Select page",
        options=["All pages"] + available_pages,
        key="kw_page_sel",
    )

    kw_data = df.copy() if kw_page_sel == "All pages" else df[df["page_label"] == kw_page_sel]

    latest_kw = kw_data[kw_data["date"] == max_date].copy()
    if latest_kw.empty and not kw_data.empty:
        latest_kw = kw_data[kw_data["date"] == kw_data["date"].max()].copy()

    if not latest_kw.empty:
        kw_display = (
            latest_kw.groupby(["page_label", "keyword_tracked"])
            .agg(
                web_rank=("organic_results_web_rank", "mean"),
                true_rank=("organic_results_true_rank", "mean"),
                search_volume=("search_volume", "first"),
            )
            .reset_index()
            .sort_values("web_rank")
            .rename(columns={
                "page_label": "Page",
                "keyword_tracked": "Keyword",
                "web_rank": "Web Rank",
                "true_rank": "True Rank",
                "search_volume": "Est. Monthly Volume",
            })
        )
        kw_display["Web Rank"] = kw_display["Web Rank"].round(1)
        kw_display["True Rank"] = kw_display["True Rank"].round(1)
        kw_display["Est. Monthly Volume"] = kw_display["Est. Monthly Volume"].astype(int)

        st.dataframe(kw_display, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(kw_display)} keywords as of {latest_kw['date'].max()}")
    else:
        st.info("No keyword data available for this selection.")

# ── Footer ────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Source: `lakehouse_production.common.seo_fact_clarity_keywords_rankings_json` · "
    f"Date range: {min(df['date'])} to {max(df['date'])} · "
    f"Device: {device} · "
    f"Total rows: {len(df):,}"
)
