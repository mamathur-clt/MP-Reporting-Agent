"""
Streamlit app: Energy Marketplace Funnel KPI Driver Analysis

Main entry point. Connects all modules: data fetch, KPI computation,
decomposition, and narrative generation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from app.config import KPIS, DRIVER_DIMENSIONS, TIME_MODES, DIMENSION_DISPLAY_NAMES, DEFAULT_CHANNELS
from app.data import fetch_session_data, default_date_range
from app.time_periods import resolve_periods
from app.kpi_engine import compute_kpi_summary, compute_funnel_table, compute_kpi
from app.decomposition import (
    decompose_all_dimensions,
    rank_top_drivers,
    analyze_initiatives,
    compute_initiative_impact,
    _initiative_label,
)
from app.narrative import generate_llm_narrative, build_chat_system_prompt, stream_chat_response, run_analyst_chat
from app.analyst_tools import (
    build_schema_context,
    build_analyst_system_prompt,
    TOOL_DEFINITIONS,
    make_tool_executor,
)
from app.finance_data import fetch_finance_daily, fetch_plan_pacing, build_funnel_summary, render_summary_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pp(delta: float) -> str:
    pp = delta * 100
    return f"{pp:+.2f}pp"


def _fmt_pct_change(delta: float, base: float) -> str:
    if base == 0:
        return "N/A"
    pct = (delta / base) * 100
    return f"{pct:+.1f}%"


def _dim_label(dim: str) -> str:
    return DIMENSION_DISPLAY_NAMES.get(dim, dim.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy MP – Funnel KPI Drivers",
    page_icon="⚡",
    layout="wide",
)

st.title("Energy Marketplace — Funnel KPI Driver Analysis")
st.caption(
    "Period-over-period decomposition of funnel KPIs. "
    "Explains what drove movement via mix-shift / rate-change analysis."
)

# ---------------------------------------------------------------------------
# Sidebar: controls
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

    st.divider()
    st.subheader("Filters")
    # Placeholders — real filters rendered after data loads
    channel_filter = st.multiselect("Marketing channel", options=[], key="channel_placeholder")
    website_filter = st.multiselect("Website", options=[], key="website_placeholder")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# Pad query start back 1 day: the step_completions CTE in the session_level_query
# uses `_date > var.start_date` (strict >), so sessions on the exact start_date
# would miss their SSN/credit completion joins without the padding.
query_start = min(curr_start, prior_start) - timedelta(days=1)
query_end = max(curr_end, prior_end)

with st.spinner("Loading session data from Databricks…"):
    df_all = fetch_session_data(str(query_start), str(query_end))

if df_all.empty:
    st.error("No data returned from Databricks. Check your date range and credentials.")
    st.stop()

available_channels = sorted(df_all["marketing_channel"].dropna().unique().tolist())
available_websites = sorted(df_all["website"].dropna().unique().tolist())

# Default to core channels, but let user override to any combination
default_ch = [c for c in DEFAULT_CHANNELS if c in available_channels]

with st.sidebar:
    channel_filter = st.multiselect(
        "Marketing channel ",
        options=available_channels,
        default=default_ch,
        help="Defaults to Paid Search, Direct, Organic, pMax. Clear all for all channels.",
    )
    website_filter = st.multiselect(
        "Website ",
        options=available_websites,
        default=[],
        help="Leave empty for all websites",
    )

# ---------------------------------------------------------------------------
# Filter and split
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
# Performance Summary (finance source of truth — top of page)
# ---------------------------------------------------------------------------
try:
    with st.spinner("Loading finance performance data…"):
        _finance_df = fetch_finance_daily()
        _plan_df = fetch_plan_pacing()

    _summary_channels = channel_filter if channel_filter else available_channels
    _summary_rows = build_funnel_summary(
        _finance_df, _plan_df, _summary_channels,
        curr_start=curr_start, curr_end=curr_end,
        prior_start=prior_start, prior_end=prior_end,
    )

    _period_label = f"{curr_start.strftime('%-m/%-d')} – {curr_end.strftime('%-m/%-d/%y')}"

    st.header("Performance Summary")
    st.caption(
        "Source: **Finance reporting queries** (source of truth for performance). "
        "Session-level driver analysis below is directional and may not tie exactly."
    )
    st.markdown(
        render_summary_html(_summary_rows, date.today(), period_label=_period_label),
        unsafe_allow_html=True,
    )
    st.divider()
except Exception as e:
    st.warning(f"Finance summary unavailable: {e}")

# ---------------------------------------------------------------------------
# Section 1: KPI Summary
# ---------------------------------------------------------------------------
st.header(f"1. {KPIS[kpi_key].name} — Period Summary")

summary = compute_kpi_summary(df_current, df_prior, kpi_key)
pct_change_str = _fmt_pct_change(summary["delta"], summary["prior_rate"])

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Current",
    f"{summary['current_rate'] * 100:.2f}%",
    delta=f"{pct_change_str} ({_fmt_pp(summary['delta'])})",
)
col2.metric("Prior", f"{summary['prior_rate'] * 100:.2f}%")
col3.metric(
    "Current volume",
    f"{int(summary['current_numerator']):,} / {int(summary['current_denominator']):,}",
)
col4.metric(
    "Prior volume",
    f"{int(summary['prior_numerator']):,} / {int(summary['prior_denominator']):,}",
)

with st.expander("Full funnel summary", expanded=False):
    funnel_df = compute_funnel_table(df_current, df_prior)
    display_df = funnel_df.copy()
    for c in ["current_rate", "prior_rate"]:
        display_df[c] = (display_df[c] * 100).round(2).astype(str) + "%"
    display_df["delta"] = (funnel_df["delta"] * 100).map(lambda x: f"{x:+.2f}pp")
    display_df["pct_change"] = (funnel_df["pct_change"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Compute decomposition + initiative impact (used by sections 2, 4, 6, 7)
# ---------------------------------------------------------------------------
# Landing page type is only meaningful when filtering to Organic traffic
_organic_only = channel_filter and all(
    c.lower() == "organic" for c in channel_filter
)
_driver_dims = [
    d for d in DRIVER_DIMENSIONS
    if d != "landing_page_type" or _organic_only
]

decomp_results = decompose_all_dimensions(df_current, df_prior, kpi_key, _driver_dims)
top_drivers = rank_top_drivers(decomp_results, n=8)

impact_current = compute_initiative_impact(df_current, kpi_key)
impact_prior = compute_initiative_impact(df_prior, kpi_key)
init_current = analyze_initiatives(df_current, kpi_key)

# ---------------------------------------------------------------------------
# Section 2: What Drove the Change (waterfall with dimensional + initiative bars)
# ---------------------------------------------------------------------------
st.header("2. What Drove the Change")

# Combine dimensional drivers + initiative impact into one chart
all_bars = []

if not top_drivers.empty:
    for _, row in top_drivers.iterrows():
        if str(row["segment"]).strip().lower() in ("unknown", "nan", "none", ""):
            continue
        all_bars.append({
            "label": f"{_dim_label(row['dimension'])}: {row['segment']}",
            "mix_pp": row["mix_effect"] * 100,
            "rate_pp": row["rate_effect"] * 100,
            "pp": row["total_contribution"] * 100,
            "type": "dimension",
        })

# Add initiative WoW change bars only if large enough to compete with dimensional drivers
if not impact_current.empty and all_bars:
    _min_dim_bar = min(abs(b["pp"]) for b in all_bars)
    _init_delta = pd.merge(
        impact_current[["initiative", "scaled_impact_on_kpi"]].rename(
            columns={"scaled_impact_on_kpi": "curr_impact"}
        ),
        impact_prior[["initiative", "scaled_impact_on_kpi"]].rename(
            columns={"scaled_impact_on_kpi": "prior_impact"}
        ) if not impact_prior.empty else pd.DataFrame(columns=["initiative", "prior_impact"]),
        on="initiative", how="left",
    ).fillna(0)
    _init_delta["wow_change"] = _init_delta["curr_impact"] - _init_delta["prior_impact"]

    for _, row in _init_delta.iterrows():
        wow_pp = row["wow_change"] * 100
        if abs(wow_pp) >= _min_dim_bar:
            all_bars.append({
                "label": f"Initiative: {row['initiative']}",
                "mix_pp": 0.0,
                "rate_pp": wow_pp,
                "pp": wow_pp,
                "type": "initiative",
            })

if all_bars:
    bars_df = pd.DataFrame(all_bars).sort_values("pp", ascending=True)

    mix_colors = ["#82e0aa" if v >= 0 else "#f1948a" for v in bars_df["mix_pp"]]
    rate_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in bars_df["rate_pp"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=bars_df["label"],
        x=bars_df["mix_pp"],
        orientation="h",
        name="Mix Effect",
        marker_color=mix_colors,
        hovertemplate="%{y}<br>Mix: %{x:+.2f}pp<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=bars_df["label"],
        x=bars_df["rate_pp"],
        orientation="h",
        name="Rate Effect",
        marker_color=rate_colors,
        text=[f"{v:+.2f}pp" for v in bars_df["pp"]],
        textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{y}<br>Rate: %{x:+.2f}pp<extra></extra>",
    ))

    total_pp = summary["delta"] * 100
    fig.update_layout(
        barmode="relative",
        title=f"Top drivers of {KPIS[kpi_key].short_name} change ({pct_change_str}, {total_pp:+.2f}pp)",
        xaxis_title="Contribution (pp)",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="gray"),
        yaxis=dict(
            tickfont=dict(size=15, color="#1a1a1a", family="Arial, sans-serif"),
        ),
        yaxis_title="",
        height=max(480, len(bars_df) * 55),
        margin=dict(l=20, r=120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each bar splits into **mix effect** (lighter — traffic share shifted) "
        "and **rate effect** (darker — conversion rate changed within that segment). "
        "Initiative bars show WoW change in scaled impact (rate only)."
    )

    with st.expander("Driver detail (mix vs rate breakdown)"):
        if not top_drivers.empty:
            detail_df = top_drivers[
                ["dimension", "segment", "mix_effect", "rate_effect", "total_contribution"]
            ].copy()
            detail_df["dimension"] = detail_df["dimension"].map(_dim_label)
            for c in ["mix_effect", "rate_effect", "total_contribution"]:
                detail_df[c] = (detail_df[c] * 100).map(lambda x: f"{x:+.2f}pp")
            detail_df.columns = ["Dimension", "Segment", "Mix Effect", "Rate Effect", "Total"]
            st.dataframe(
                detail_df.sort_values("Total", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("No meaningful drivers found for this KPI and period.")

# ---------------------------------------------------------------------------
# Section 3: Dimension detail tables (always includes all dimensions)
# ---------------------------------------------------------------------------
st.header("3. Dimension Detail")

_all_decomp = decompose_all_dimensions(df_current, df_prior, kpi_key, DRIVER_DIMENSIONS)
visible_dims = [d for d in DRIVER_DIMENSIONS if d in _all_decomp]
dim_tabs = st.tabs([_dim_label(d) for d in visible_dims])
for tab, dim in zip(dim_tabs, visible_dims):
    with tab:
        detail = _all_decomp[dim].copy()
        for c in ["curr_rate", "prior_rate"]:
            if c in detail.columns:
                detail[c] = (detail[c] * 100).round(2).astype(str) + "%"
        for c in ["curr_weight", "prior_weight"]:
            if c in detail.columns:
                detail[c] = (detail[c] * 100).round(1).astype(str) + "%"
        for c in ["mix_effect", "rate_effect", "total_contribution"]:
            if c in detail.columns:
                detail[c] = (detail[c] * 100).map(lambda x: f"{x:+.2f}pp")
        col_rename = {
            "segment": "Segment",
            "curr_den": "Current Vol",
            "prior_den": "Prior Vol",
            "curr_weight": "Current Share",
            "prior_weight": "Prior Share",
            "curr_rate": "Current Rate",
            "prior_rate": "Prior Rate",
            "mix_effect": "Mix Effect",
            "rate_effect": "Rate Effect",
            "total_contribution": "Total",
        }
        detail = detail.rename(columns={k: v for k, v in col_rename.items() if k in detail.columns})
        st.dataframe(detail, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 4: Initiative Impact
# ---------------------------------------------------------------------------
st.header("4. Initiative Impact")
st.caption(
    "Did initiatives help or hurt the KPI this period? "
    "Each initiative's lift (model rate − holdout rate) is scaled by its "
    "share of traffic to show the actual contribution to the all-in KPI."
)

if not impact_current.empty:
    total_impact_curr = impact_current["scaled_impact_on_kpi"].sum()
    total_impact_prior = impact_prior["scaled_impact_on_kpi"].sum() if not impact_prior.empty else 0.0
    impact_change = total_impact_curr - total_impact_prior

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric(
        "Total initiative contribution",
        _fmt_pp(total_impact_curr),
        help="Sum of all model/FMP lifts scaled by traffic share",
    )
    ic2.metric("Prior period contribution", _fmt_pp(total_impact_prior))
    ic3.metric(
        "Change in initiative contribution",
        _fmt_pp(impact_change),
        delta=_fmt_pp(impact_change),
    )

    chart_impact = impact_current.sort_values("scaled_impact_on_kpi", ascending=True)
    fig_init = go.Figure(go.Bar(
        y=chart_impact["initiative"],
        x=(chart_impact["scaled_impact_on_kpi"] * 100).tolist(),
        orientation="h",
        marker_color=[
            "#3498db" if v >= 0 else "#e67e22"
            for v in chart_impact["scaled_impact_on_kpi"]
        ],
        text=[f"{v * 100:+.3f}pp" for v in chart_impact["scaled_impact_on_kpi"]],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig_init.update_layout(
        title=f"Initiative contribution to {KPIS[kpi_key].short_name} (current period)",
        xaxis_title="Scaled impact on all-in KPI (pp)",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="gray"),
        yaxis_title="",
        height=max(300, len(chart_impact) * 60),
        margin=dict(l=20, r=100),
        showlegend=False,
    )
    st.plotly_chart(fig_init, use_container_width=True)

    st.subheader("Period-over-period comparison")

    merged_impact = pd.merge(
        impact_current.rename(columns={
            "model_rate": "curr_model_rate", "holdout_rate": "curr_holdout_rate",
            "lift": "curr_lift", "model_share": "curr_share",
            "scaled_impact_on_kpi": "curr_impact", "model_sessions": "curr_sessions",
        }),
        impact_prior.rename(columns={
            "model_rate": "prior_model_rate", "holdout_rate": "prior_holdout_rate",
            "lift": "prior_lift", "model_share": "prior_share",
            "scaled_impact_on_kpi": "prior_impact", "model_sessions": "prior_sessions",
        }),
        on="initiative", how="outer",
    ).fillna(0)

    merged_impact["lift_change"] = merged_impact["curr_lift"] - merged_impact["prior_lift"]
    merged_impact["impact_change"] = merged_impact["curr_impact"] - merged_impact["prior_impact"]

    display_impact = pd.DataFrame({
        "Initiative": merged_impact["initiative"],
        "Curr Model Rate": (merged_impact["curr_model_rate"] * 100).round(2).astype(str) + "%",
        "Curr Holdout Rate": (merged_impact["curr_holdout_rate"] * 100).round(2).astype(str) + "%",
        "Curr Lift": merged_impact["curr_lift"].map(lambda x: f"{x * 100:+.2f}pp"),
        "Prior Lift": merged_impact["prior_lift"].map(lambda x: f"{x * 100:+.2f}pp"),
        "Lift Change": merged_impact["lift_change"].map(lambda x: f"{x * 100:+.2f}pp"),
        "Curr Share": (merged_impact["curr_share"] * 100).round(1).astype(str) + "%",
        "Curr Impact": merged_impact["curr_impact"].map(lambda x: f"{x * 100:+.3f}pp"),
        "Prior Impact": merged_impact["prior_impact"].map(lambda x: f"{x * 100:+.3f}pp"),
        "Impact Change": merged_impact["impact_change"].map(lambda x: f"{x * 100:+.3f}pp"),
    })
    st.dataframe(display_impact, use_container_width=True, hide_index=True)

    st.caption(
        "**Lift** = model rate − holdout rate. "
        "**Share** = initiative's share of the KPI denominator. "
        "**Scaled Impact** = lift × share = how much the initiative adds to the all-in KPI vs holdout. "
        "**Impact Change** = did the initiative's contribution improve or worsen vs prior period?"
    )
else:
    st.info("No initiative data available for this period.")

# ---------------------------------------------------------------------------
# Section 5: Cart Conversion decomposition
# ---------------------------------------------------------------------------
if kpi_key in ("Cart Conversion", "VC"):
    st.header("5. Cart Conversion Decomposition")
    st.caption(
        "Cart Conversion = SSN Submit Rate × Conversion After Credit. "
        "Isolates pre-credit friction from post-credit eligibility."
    )
    sub_kpis = ["SSN Submit Rate", "Conversion After Credit"]
    cols = st.columns(len(sub_kpis))
    for col, sk in zip(cols, sub_kpis):
        with col:
            s = compute_kpi_summary(df_current, df_prior, sk)
            pct_ch = _fmt_pct_change(s["delta"], s["prior_rate"])
            st.metric(
                KPIS[sk].name,
                f"{s['current_rate'] * 100:.2f}%",
                delta=f"{pct_ch} ({_fmt_pp(s['delta'])})",
            )

# ---------------------------------------------------------------------------
# Section 6: LLM-generated summary
# ---------------------------------------------------------------------------
st.header("6. Weekly Review Summary")
st.caption("AI-generated commentary — review before sharing.")

with st.spinner("Generating summary with GPT-4o…"):
    narrative = generate_llm_narrative(
        summary, top_drivers, init_current,
        initiative_impact=impact_current if not impact_current.empty else None,
        initiative_impact_prior=impact_prior if not impact_prior.empty else None,
    )

st.markdown(narrative)

# ---------------------------------------------------------------------------
# Section 7: Ask the Analyst (with data access)
# ---------------------------------------------------------------------------
st.header("7. Ask the Analyst")
st.caption(
    "Chat with GPT-4o — it can query the loaded session and finance data "
    "or run SQL against Databricks to answer ad-hoc questions."
)

# Build base analysis context (reuses existing prompt builder)
_funnel_text = ""
try:
    _funnel_for_chat = compute_funnel_table(df_current, df_prior)
    _fc = _funnel_for_chat.copy()
    _fc["current_rate"] = (_funnel_for_chat["current_rate"] * 100).round(2).astype(str) + "%"
    _fc["prior_rate"] = (_funnel_for_chat["prior_rate"] * 100).round(2).astype(str) + "%"
    _fc["delta"] = (_funnel_for_chat["delta"] * 100).map(lambda x: f"{x:+.2f}pp")
    _funnel_text = _fc[["kpi", "current_rate", "prior_rate", "delta"]].to_string(index=False)
except Exception:
    pass

_chat_system_prompt = build_chat_system_prompt(
    summary, top_drivers, init_current, funnel_table_text=_funnel_text,
    initiative_impact=impact_current if not impact_current.empty else None,
    initiative_impact_prior=impact_prior if not impact_prior.empty else None,
)

# Augment with data-access tools
_analyst_finance = None
try:
    _analyst_finance = _finance_df  # may not exist if finance fetch failed
except NameError:
    pass

_schema_ctx = build_schema_context(df_all, _analyst_finance)
_filter_parts = []
if channel_filter:
    _filter_parts.append(f"Channels: {', '.join(channel_filter)}")
if website_filter:
    _filter_parts.append(f"Websites: {', '.join(website_filter)}")
_analyst_prompt = build_analyst_system_prompt(
    _chat_system_prompt,
    _schema_ctx,
    current_filters="; ".join(_filter_parts) if _filter_parts else "",
)
_tool_executor = make_tool_executor(df_all, _analyst_finance)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Replay previous messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        for tr in msg.get("tool_results", []):
            with st.expander(f"Data: {tr['explanation']}", expanded=False):
                if isinstance(tr.get("result_obj"), pd.DataFrame):
                    st.dataframe(tr["result_obj"], use_container_width=True, hide_index=True)
                elif tr.get("result_str"):
                    st.code(tr["result_str"])
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the data…"):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            try:
                final_text, tool_results = run_analyst_chat(
                    st.session_state.chat_messages,
                    _analyst_prompt,
                    TOOL_DEFINITIONS,
                    _tool_executor,
                )
            except Exception as e:
                final_text = f"Error: {e}"
                tool_results = []

        for tr in tool_results:
            with st.expander(f"Data: {tr['explanation']}", expanded=True):
                if isinstance(tr.get("result_obj"), pd.DataFrame):
                    st.dataframe(
                        tr["result_obj"], use_container_width=True, hide_index=True,
                    )
                elif tr.get("result_str"):
                    st.code(tr["result_str"])

        st.markdown(final_text)
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": final_text,
            "tool_results": [
                {
                    "explanation": tr["explanation"],
                    "result_str": tr["result_str"],
                    "result_obj": tr["result_obj"],
                }
                for tr in tool_results
            ],
        })

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Data range queried: {query_start} to {query_end} · "
    f"Total sessions loaded: {len(df_all):,} · "
    f"Current period sessions: {len(df_current):,} · "
    f"Prior period sessions: {len(df_prior):,}"
)
