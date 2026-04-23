"""
Paid Search deep-dive tab (Phase 4).

Auto-filters to ``marketing_channel IN ('Paid Search', 'pMax')`` — the
sidebar channel filter is overridden on this tab. Sections follow the
PRD:

  1. Paid Pacing Summary           — finance source-of-truth, filtered
  2. Paid Revenue Waterfall        — Plan → Pacing, sequential substitution
  3. Campaign Bucket Performance   — period-over-period bucket table
  4. Campaign Bucket VC Decomp     — VC driver waterfall: mix-vs-perf by bucket
  5. Campaign Bucket Drill-Down    — per-bucket trend, funnel, efficiency

Two time systems live on this tab, mirroring the Organic tab:

  • Sidebar window  — drives Sections 1–2 (Finance / Plan, monthly
                      Pacing vs Plan).
  • Paid window     — drives Sections 3–5 (Paid Query data). Tab-local
                      WoW / MoM MTD / Custom picker, so the bucket
                      analysis can use a different cadence than the
                      monthly finance view above it.

The Paid Query is fetched ONCE per render, covering both the current and
prior windows, and all three paid sections filter off that single frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.app_context import AppContext
from app.finance_data import build_funnel_summary, render_summary_html
from app.paid_search_data import (
    CAMPAIGN_BUCKETS,
    BucketVCDecompResult,
    aggregate_total_period,
    bucket_phone_funnel,
    bucket_vc_decomposition,
    compare_bucket_periods,
    daily_bucket_trend,
    fetch_paid_for_windows,
)
from app.time_periods import resolve_periods
from app.waterfall import render_waterfall_section


# Channel definitions — these are the UI-level labels used in the
# session-level DataFrame and the finance pacing table.
PAID_CHANNELS = ["Paid Search", "pMax"]

# Palette shared with the Organic tab so charts across deep-dives look
# consistent. Buckets get deterministic colors based on their index in
# `CAMPAIGN_BUCKETS`.
_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#4C78A8", "#F58518",
]

_BAR_UP = "#27ae60"
_BAR_DOWN = "#e74c3c"
_BAR_TOTAL = "#2c3e50"


# ---------------------------------------------------------------------------
# Tab-local period resolution
# ---------------------------------------------------------------------------


@dataclass
class PaidPeriod:
    """Resolved Paid-tab comparison window."""

    mode: str
    curr_start: date
    curr_end: date
    prior_start: date
    prior_end: date

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


_TAB_TIME_MODES = ["WoW", "MoM MTD", "Custom"]


def _resolve_paid_tab_period(mode: str, today: date | None = None) -> tuple[date, date, date, date]:
    today = today or date.today()
    if mode == "MoM MTD":
        return resolve_periods("MoM", ref_date=today)
    if mode == "WoW":
        return resolve_periods("WoW", ref_date=today)
    raise ValueError(f"Use _render_period_picker for custom mode (mode={mode!r}).")


def _render_period_picker() -> PaidPeriod:
    """Render the tab-local comparison-window picker (mirrors Organic tab)."""
    with st.container():
        cols = st.columns([1, 3])
        with cols[0]:
            mode = st.radio(
                "Paid window",
                options=_TAB_TIME_MODES,
                index=0,
                key="paid_tab_mode",
                help=(
                    "Drives Sections 3–5 (Paid Query bucket data). Sidebar "
                    "continues to drive Sections 1–2 (finance pacing / waterfall)."
                ),
            )
        with cols[1]:
            if mode == "Custom":
                today = date.today()
                c1, c2 = st.columns(2)
                with c1:
                    curr_range = st.date_input(
                        "Current",
                        value=(today - timedelta(days=7), today - timedelta(days=1)),
                        key="paid_tab_custom_curr",
                    )
                with c2:
                    prior_range = st.date_input(
                        "Prior",
                        value=(today - timedelta(days=14), today - timedelta(days=8)),
                        key="paid_tab_custom_prior",
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
                curr_start, curr_end, prior_start, prior_end = _resolve_paid_tab_period(mode)
                st.markdown(
                    f"**Current:** `{curr_start} → {curr_end}` ({(curr_end - curr_start).days + 1}d) · "
                    f"**Prior:** `{prior_start} → {prior_end}` ({(prior_end - prior_start).days + 1}d)"
                )

    return PaidPeriod(
        mode=mode,
        curr_start=curr_start, curr_end=curr_end,
        prior_start=prior_start, prior_end=prior_end,
    )


# ---------------------------------------------------------------------------
# Sections 1 & 2 — finance-driven (sidebar window)
# ---------------------------------------------------------------------------


def _render_sidebar_note() -> None:
    st.info(
        "**Paid Search deep-dive tab.** The sidebar channel filter is "
        "overridden — every section reports on "
        "`marketing_channel IN ('Paid Search', 'pMax')`. Sections 1–2 "
        "(Pacing / Revenue Waterfall) follow the **sidebar** window; "
        "Sections 3–5 (Campaign Bucket analysis) follow the tab-local "
        "**Paid window** picker."
    )


def _render_pacing_summary(ctx: AppContext) -> None:
    st.header("1. Paid Pacing Summary")
    st.caption("Sidebar-driven. Source: finance pacing (`rpt_texas_daily_pacing`).")
    if ctx.finance_df is None or ctx.plan_df is None:
        st.warning("Finance pacing data unavailable.")
        return

    try:
        rows = build_funnel_summary(
            ctx.finance_df,
            ctx.plan_df,
            PAID_CHANNELS,
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
        st.warning(f"Paid pacing summary unavailable: {e}")


def _render_waterfall(ctx: AppContext) -> None:
    st.header("2. Paid Search Revenue Waterfall (vs Plan)")
    st.caption(
        "Sidebar-driven. Pacing vs Plan revenue gap for Paid Search + pMax, "
        "decomposed via sequential substitution across Sessions → Site RR → "
        "Site Conversion → Phone GCV/Order → Cart RR → Cart Conversion → "
        "Cart GCV/Order."
    )
    result = render_waterfall_section(
        ctx.plan_df,
        PAID_CHANNELS,
        chart_title="Paid Search Revenue Waterfall — Plan → Pacing",
        caption="Source: `rpt_texas_daily_pacing` filtered to Paid Search + PMAX plan channels.",
    )
    if result is not None:
        ctx.cache["paid_waterfall"] = result


# ---------------------------------------------------------------------------
# Section 3 — Campaign Bucket Performance Table
# ---------------------------------------------------------------------------
#
# The table has 10 metric columns. For each, we render the current-period
# absolute value plus a color-coded fractional delta. The palette mirrors
# `finance_data.render_summary_html` so exec-facing tables look the same
# across tabs.

_BUCKET_COLOR_GREEN = "background-color:#c6efce;color:#006100;"
_BUCKET_COLOR_RED = "background-color:#ffc7ce;color:#9c0006;"
_BUCKET_COLOR_YELLOW = "background-color:#ffeb9c;color:#9c6500;"
_BUCKET_PCT_THRESHOLD = 0.025  # ±2.5% band → yellow

_BUCKET_TABLE_STYLE = (
    "border-collapse:collapse;width:100%;font-family:Arial,sans-serif;"
    "font-size:13px;"
)
_BUCKET_HEADER_ROW_STYLE = "background:#2c3e50;color:white;text-align:center;"
_BUCKET_HEADER_CELL_STYLE = "padding:8px 10px;border:1px solid #bbb;font-weight:700;"
_BUCKET_BODY_CELL_STYLE = (
    "padding:6px 10px;border:1px solid #ddd;text-align:right;font-weight:600;"
)
_BUCKET_BODY_LABEL_STYLE = (
    "padding:6px 12px;border:1px solid #ddd;font-weight:700;text-align:left;"
)


def _fmt_int(v: float) -> str:
    return f"{int(round(v)):,}" if pd.notna(v) else "—"


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}" if pd.notna(v) else "—"


def _fmt_money_precise(v: float) -> str:
    return f"${v:,.2f}" if pd.notna(v) else "—"


def _fmt_pct_value(v: float) -> str:
    return f"{v * 100:.2f}%" if pd.notna(v) else "—"


def _fmt_pct_change(v: float | None) -> str:
    return f"{v * 100:+.1f}%" if v is not None and pd.notna(v) else "—"


def _fmt_float(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if pd.notna(v) else "—"


def _delta_bg(delta: float | None, *, inverted: bool = False) -> str:
    """Return the inline CSS background for a delta cell. `inverted=True`
    flips the sign check — useful for cost-like metrics where a decrease
    is a win (CPC, cost/order).
    """
    if delta is None or pd.isna(delta):
        return ""
    if abs(delta) < _BUCKET_PCT_THRESHOLD:
        return _BUCKET_COLOR_YELLOW
    positive = (delta > 0) != inverted
    return _BUCKET_COLOR_GREEN if positive else _BUCKET_COLOR_RED


def _render_bucket_table(compare_df: pd.DataFrame) -> None:
    """
    Render the period-over-period bucket table from `compare_bucket_periods`.

    12 columns per row: bucket label, then six "value + Δ%" pairs
    covering the headline marketing + funnel metrics. The first three
    columns (Impressions / Clicks / CTR / Cost / CPC) capture the
    auction-level story; the next three (Sessions / VC / Revenue)
    capture the funnel-level story. Buckets are ordered by registry
    (brand-first), with unknown buckets appended at the end.

    Rendered as raw HTML so the exec table copy-pastes cleanly into
    Slack and retains its styling — matches the `finance_data` pattern.
    """
    if compare_df is None or compare_df.empty:
        st.info("No campaign bucket data for the current window.")
        return

    header_cells = [
        ("Campaign Bucket", "text-align:left;min-width:140px;"),
        ("Impr", ""), ("Δ", ""),
        ("Clicks", ""), ("Δ", ""),
        ("CTR", ""), ("Δ", ""),
        ("Cost", ""), ("Δ", ""),
        ("CPC", ""), ("Δ", ""),
        ("Sessions", ""), ("Δ", ""),
        ("Orders", ""), ("Δ", ""),
        ("VC", ""), ("Δ", ""),
        ("Revenue", ""), ("Δ", ""),
    ]
    header_html = "".join(
        f'<th style="{_BUCKET_HEADER_CELL_STYLE}{extra}">{lbl}</th>'
        for lbl, extra in header_cells
    )
    thead = f'<thead><tr style="{_BUCKET_HEADER_ROW_STYLE}">{header_html}</tr></thead>'

    body_rows: list[str] = []
    for _, row in compare_df.iterrows():
        cells: list[tuple[str, str]] = [
            (str(row["campaign_bucket"]), _BUCKET_BODY_LABEL_STYLE),
        ]

        def _pair(val_str: str, delta_val: float | None, *, inverted: bool = False) -> None:
            cells.append((val_str, _BUCKET_BODY_CELL_STYLE))
            cells.append((
                _fmt_pct_change(delta_val),
                _BUCKET_BODY_CELL_STYLE + _delta_bg(delta_val, inverted=inverted),
            ))

        _pair(_fmt_int(row["impressioncount_curr"]), row["impressioncount_delta_pct"])
        _pair(_fmt_int(row["clickcount_curr"]), row["clickcount_delta_pct"])
        _pair(_fmt_pct_value(row["ctr_curr"]), row["ctr_delta_pct"])
        _pair(_fmt_money(row["cost_curr"]), row["cost_delta_pct"], inverted=True)
        _pair(_fmt_money_precise(row["cpc_curr"]), row["cpc_delta_pct"], inverted=True)
        _pair(_fmt_int(row["sessions_curr"]), row["sessions_delta_pct"])
        _pair(_fmt_int(row["total_orders_curr"]), row["total_orders_delta_pct"])
        _pair(_fmt_pct_value(row["vc_curr"]), row["vc_delta_pct"])
        _pair(_fmt_money(row["est_rev_curr"]), row["est_rev_delta_pct"], inverted=False)

        cells_html = "".join(f'<td style="{style}">{val}</td>' for val, style in cells)
        body_rows.append(f"<tr>{cells_html}</tr>")

    html = (
        f'<div style="overflow-x:auto;">'
        f'<table style="{_BUCKET_TABLE_STYLE}">'
        f'{thead}<tbody>{"".join(body_rows)}</tbody>'
        f"</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_bucket_performance(
    period: PaidPeriod, paid_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Section 3 — Campaign Bucket Performance Table.

    Returns the comparison DataFrame so downstream sections don't need
    to recompute it.
    """
    st.header("3. Campaign Bucket Performance")
    st.caption(
        f"Period-over-period bucket rollup from `docs/Paid Query.sql` · "
        f"**{period.window_label}** ({period.curr_days}d each). "
        f"Green = +>2.5%, red = <-2.5%, yellow = within band. Cost & CPC "
        f"deltas are color-inverted (lower = better)."
    )

    compare_df = compare_bucket_periods(
        paid_df,
        period.curr_start, period.curr_end,
        period.prior_start, period.prior_end,
    )
    if compare_df is None or compare_df.empty:
        st.info("No paid search campaign data returned for this window.")
        return pd.DataFrame()

    # Headline summary cards — totals across buckets.
    totals_curr = aggregate_total_period(
        paid_df, period.curr_start, period.curr_end
    )
    totals_prior = aggregate_total_period(
        paid_df, period.prior_start, period.prior_end
    )

    def _delta(curr: float, prior: float) -> str | None:
        if not prior:
            return None
        return f"{(curr - prior) / prior * 100:+.1f}%"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost", _fmt_money(totals_curr["cost"]),
              delta=_delta(totals_curr["cost"], totals_prior["cost"]),
              delta_color="inverse")
    c2.metric("Clicks", _fmt_int(totals_curr["clickcount"]),
              delta=_delta(totals_curr["clickcount"], totals_prior["clickcount"]))
    c3.metric("Sessions", _fmt_int(totals_curr["sessions"]),
              delta=_delta(totals_curr["sessions"], totals_prior["sessions"]))
    c4.metric("Revenue", _fmt_money(totals_curr["est_rev"]),
              delta=_delta(totals_curr["est_rev"], totals_prior["est_rev"]))

    _render_bucket_table(compare_df)

    # Top movers callout — ranks buckets by absolute revenue change.
    movers = compare_df.assign(
        abs_rev_delta=compare_df["est_rev_delta"].abs()
    ).sort_values("abs_rev_delta", ascending=False).head(3)
    if not movers.empty and movers["abs_rev_delta"].iloc[0] > 0:
        bits: list[str] = []
        for _, r in movers.iterrows():
            direction = "↑" if r["est_rev_delta"] > 0 else "↓"
            bits.append(
                f"**{r['campaign_bucket']}** {direction} "
                f"{_fmt_money(r['est_rev_delta'])} "
                f"({_fmt_pct_change(r['est_rev_delta_pct'])})"
            )
        st.markdown(
            "**Top movers (revenue):** " + " · ".join(bits)
        )

    with st.expander("Full bucket detail (all metrics)"):
        detail = pd.DataFrame({
            "Bucket": compare_df["campaign_bucket"],
            "Impressions (Curr)": compare_df["impressioncount_curr"].map(_fmt_int),
            "Impressions (Prior)": compare_df["impressioncount_prior"].map(_fmt_int),
            "Clicks (Curr)": compare_df["clickcount_curr"].map(_fmt_int),
            "Clicks (Prior)": compare_df["clickcount_prior"].map(_fmt_int),
            "CTR (Curr)": compare_df["ctr_curr"].map(_fmt_pct_value),
            "CTR (Prior)": compare_df["ctr_prior"].map(_fmt_pct_value),
            "Cost (Curr)": compare_df["cost_curr"].map(_fmt_money),
            "Cost (Prior)": compare_df["cost_prior"].map(_fmt_money),
            "CPC (Curr)": compare_df["cpc_curr"].map(_fmt_money_precise),
            "CPC (Prior)": compare_df["cpc_prior"].map(_fmt_money_precise),
            "Sessions (Curr)": compare_df["sessions_curr"].map(_fmt_int),
            "Sessions (Prior)": compare_df["sessions_prior"].map(_fmt_int),
            "Cart Starts (Curr)": compare_df["cart_starts_curr"].map(_fmt_int),
            "Orders (Curr)": compare_df["total_orders_curr"].map(_fmt_int),
            "Orders (Prior)": compare_df["total_orders_prior"].map(_fmt_int),
            "VC (Curr)": compare_df["vc_curr"].map(_fmt_pct_value),
            "VC (Prior)": compare_df["vc_prior"].map(_fmt_pct_value),
            "CPO (Curr)": compare_df["cost_per_order_curr"].map(_fmt_money_precise),
            "CPO (Prior)": compare_df["cost_per_order_prior"].map(_fmt_money_precise),
            "ROAS (Curr)": compare_df["roas_curr"].map(lambda v: _fmt_float(v, 2)),
            "ROAS (Prior)": compare_df["roas_prior"].map(lambda v: _fmt_float(v, 2)),
            "Revenue (Curr)": compare_df["est_rev_curr"].map(_fmt_money),
            "Revenue (Prior)": compare_df["est_rev_prior"].map(_fmt_money),
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

    return compare_df


# ---------------------------------------------------------------------------
# Section 4 — Campaign Bucket VC Decomposition Waterfall
# ---------------------------------------------------------------------------


def _build_vc_decomp_figure(
    result: BucketVCDecompResult, title: str
) -> go.Figure:
    """
    Plotly waterfall chart decomposing the portfolio VC delta into
    per-bucket mix + performance contributions (in percentage points).

    Prior VC → per-bucket (mix + perf) bars → Current VC.
    Buckets ordered by |total_impact_pp| descending.
    """
    rows = sorted(result.rows, key=lambda r: abs(r.total_impact_pp), reverse=True)
    material = [r for r in rows if abs(r.total_impact_pp) >= 0.001]
    if not material:
        material = rows

    labels = ["Prior VC"] + [r.campaign_bucket for r in material] + ["Current VC"]
    values_pp = (
        [result.prior_vc_total * 100.0]
        + [r.total_impact_pp for r in material]
        + [result.curr_vc_total * 100.0]
    )
    measure = ["absolute"] + ["relative"] * len(material) + ["total"]

    gap_pp = result.vc_delta_pp
    text: list[str] = []
    for m, v in zip(measure, values_pp):
        if m in ("absolute", "total"):
            text.append(f"<b>{v:.2f}%</b>")
            continue
        sign = "+" if v >= 0 else ""
        pct_of_gap = f"<br>({v / gap_pp * 100:+.0f}% of Δ)" if gap_pp else ""
        text.append(f"<b>{sign}{v:.3f} pp</b>{pct_of_gap}")

    fig = go.Figure(go.Waterfall(
        x=labels, y=values_pp, measure=measure, text=text,
        textposition="outside",
        textfont=dict(size=12, color="#1a1a1a"),
        connector=dict(line=dict(color="rgb(160,160,160)", width=1, dash="dot")),
        increasing=dict(marker=dict(color=_BAR_UP, line=dict(color="#145a32", width=1))),
        decreasing=dict(marker=dict(color=_BAR_DOWN, line=dict(color="#7b241c", width=1))),
        totals=dict(marker=dict(color=_BAR_TOTAL, line=dict(color="#17202a", width=1))),
        width=0.55,
    ))

    total_min = min(result.prior_vc_total, result.curr_vc_total) * 100.0
    total_max = max(result.prior_vc_total, result.curr_vc_total) * 100.0
    driver_span = max(
        total_max - total_min,
        max((abs(r.total_impact_pp) for r in material), default=0.0),
    )
    y_floor = max(0.0, total_min - driver_span * 0.5)
    y_ceiling = total_max + driver_span * 0.7

    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#1a1a1a"),
                    x=0.02, xanchor="left"),
        height=540,
        margin=dict(l=80, r=40, t=70, b=130),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        yaxis=dict(
            title=dict(text="Visit Conversion (%)", font=dict(size=13)),
            tickformat=".2f", ticksuffix="%",
            range=[y_floor, y_ceiling],
            gridcolor="rgba(0,0,0,0.08)", zeroline=False,
        ),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=13, color="#1a1a1a",
                          family="Arial Black, Arial, sans-serif"),
            showgrid=False,
        ),
    )
    fig.add_hline(
        y=result.prior_vc_total * 100.0,
        line=dict(color="rgba(44,62,80,0.35)", width=1, dash="dash"),
    )
    return fig


def _build_vc_decomp_stacked_figure(
    result: BucketVCDecompResult, title: str
) -> go.Figure:
    """
    Horizontal stacked bar chart showing mix vs perf split per bucket,
    ranked by |total_impact|. Gives a complementary view to the
    waterfall by breaking each bucket into its two components.
    """
    rows = sorted(result.rows, key=lambda r: abs(r.total_impact_pp), reverse=True)
    material = [r for r in rows if abs(r.total_impact_pp) >= 0.001]
    if not material:
        material = rows

    buckets = [r.campaign_bucket for r in material]
    mix_vals = [r.mix_impact_pp for r in material]
    perf_vals = [r.perf_impact_pp for r in material]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=buckets, x=mix_vals, orientation="h",
        name="Mix Effect (share shift)",
        marker_color="#3498db",
        hovertemplate="%{y}<br>Mix: %{x:+.3f} pp<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=buckets, x=perf_vals, orientation="h",
        name="Perf Effect (VC change)",
        marker_color="#e67e22",
        hovertemplate="%{y}<br>Perf: %{x:+.3f} pp<extra></extra>",
    ))
    fig.update_layout(
        barmode="relative",
        title=dict(text=title, font=dict(size=15, color="#1a1a1a"),
                    x=0.02, xanchor="left"),
        height=max(300, 40 * len(material) + 120),
        margin=dict(l=140, r=40, t=60, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        xaxis=dict(
            title="pp contribution to VC delta",
            tickformat="+.3f", ticksuffix=" pp",
            gridcolor="rgba(0,0,0,0.08)", zeroline=True,
            zerolinecolor="rgba(0,0,0,0.3)", zerolinewidth=1,
        ),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _render_bucket_vc_waterfall(
    period: PaidPeriod, paid_df: pd.DataFrame
) -> None:
    st.header("4. Campaign Bucket VC Decomposition")
    st.caption(
        "Period-over-period **Visit Conversion** change decomposed by "
        "campaign bucket into mix and performance effects. "
        "`mix = (Δ session_share) × prior_vc`, "
        "`perf = curr_share × (Δ vc)`. "
        "Per-bucket contributions sum exactly to the total VC delta."
    )

    result = bucket_vc_decomposition(
        paid_df,
        period.curr_start, period.curr_end,
        period.prior_start, period.prior_end,
    )
    if result is None or not result.rows:
        st.info("Not enough session data to build the VC decomposition.")
        return

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Prior VC",
        f"{result.prior_vc_total * 100:.2f}%",
    )
    c2.metric(
        "Current VC",
        f"{result.curr_vc_total * 100:.2f}%",
        delta=f"{result.vc_delta_pp:+.3f} pp",
    )
    mix_total_pp = sum(r.mix_impact_pp for r in result.rows)
    perf_total_pp = sum(r.perf_impact_pp for r in result.rows)
    c3.metric("Total Mix Effect", f"{mix_total_pp:+.3f} pp")
    c4.metric("Total Perf Effect", f"{perf_total_pp:+.3f} pp")

    # Waterfall chart
    fig = _build_vc_decomp_figure(
        result,
        title=f"VC Driver Waterfall by Campaign Bucket · {period.window_label}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Reconciliation check
    reconciled_pp = sum(r.total_impact_pp for r in result.rows)
    residual_pp = result.vc_delta_pp - reconciled_pp
    if abs(residual_pp) > 0.01:
        st.caption(
            f"Waterfall residual of {residual_pp:+.4f} pp — impacts do not "
            "fully reconcile to the VC delta (likely from buckets with "
            "zero sessions in both periods being skipped)."
        )

    # Narrative callout
    positives = [r for r in result.rows if r.total_impact_pp > 0]
    negatives = [r for r in result.rows if r.total_impact_pp < 0]
    pieces: list[str] = []
    if result.vc_delta_pp >= 0:
        pieces.append(
            f"Portfolio VC up **{result.vc_delta_pp:+.3f} pp** vs prior "
            f"({result.prior_vc_total * 100:.2f}% → {result.curr_vc_total * 100:.2f}%)."
        )
    else:
        pieces.append(
            f"Portfolio VC down **{result.vc_delta_pp:.3f} pp** vs prior "
            f"({result.prior_vc_total * 100:.2f}% → {result.curr_vc_total * 100:.2f}%)."
        )
    if negatives:
        worst = min(negatives, key=lambda r: r.total_impact_pp)
        pieces.append(
            f"Biggest drag: **{worst.campaign_bucket}** "
            f"({worst.total_impact_pp:+.3f} pp — "
            f"mix {worst.mix_impact_pp:+.3f}, perf {worst.perf_impact_pp:+.3f})."
        )
    if positives:
        best = max(positives, key=lambda r: r.total_impact_pp)
        pieces.append(
            f"Biggest tailwind: **{best.campaign_bucket}** "
            f"({best.total_impact_pp:+.3f} pp — "
            f"mix {best.mix_impact_pp:+.3f}, perf {best.perf_impact_pp:+.3f})."
        )
    st.markdown(" ".join(pieces))

    # Stacked mix / perf bar chart
    stacked_fig = _build_vc_decomp_stacked_figure(
        result,
        title="Mix vs Performance Effect by Bucket",
    )
    st.plotly_chart(stacked_fig, use_container_width=True)

    # Detail table
    frame = result.to_frame()
    detail = pd.DataFrame({
        "Bucket": frame["campaign_bucket"],
        "Curr Sessions": frame["curr_sessions"].map(_fmt_int),
        "Prior Sessions": frame["prior_sessions"].map(_fmt_int),
        "Curr Share": frame["curr_share"].map(_fmt_pct_value),
        "Prior Share": frame["prior_share"].map(_fmt_pct_value),
        "Curr VC": frame["curr_vc"].map(_fmt_pct_value),
        "Prior VC": frame["prior_vc"].map(_fmt_pct_value),
        "Curr Orders": frame["curr_orders"].map(_fmt_int),
        "Prior Orders": frame["prior_orders"].map(_fmt_int),
        "Mix (pp)": frame["mix_impact_pp"].map(lambda v: f"{v:+.3f}"),
        "Perf (pp)": frame["perf_impact_pp"].map(lambda v: f"{v:+.3f}"),
        "Net (pp)": frame["total_impact_pp"].map(lambda v: f"{v:+.3f}"),
        "% of Δ": frame["total_impact_pp"].map(
            lambda v: f"{v / result.vc_delta_pp * 100:+.0f}%"
                      if result.vc_delta_pp else "—"
        ),
    })
    with st.expander("Per-bucket mix / performance decomposition"):
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.caption(
            "Method: counterfactual decomposition. "
            "`mix` = (Δ session share) × prior bucket VC — measures "
            "the VC impact of session traffic shifting between buckets. "
            "`perf` = current share × (Δ bucket VC) — measures the VC "
            "impact of conversion changing within each bucket. "
            "Sum across all buckets reconciles to the total VC delta."
        )


# ---------------------------------------------------------------------------
# Section 5 — Campaign Bucket Drill-Down
# ---------------------------------------------------------------------------


def _render_bucket_drilldown(
    period: PaidPeriod,
    paid_df: pd.DataFrame,
    compare_df: pd.DataFrame,
) -> None:
    st.header("5. Campaign Bucket Drill-Down")
    st.caption(
        "Pick a bucket to see its daily trend (impressions / clicks / "
        "cost / sessions), full cart + phone funnels, and efficiency "
        "metrics (CPC, cost/order, ROAS) over the current window."
    )

    if compare_df is None or compare_df.empty:
        st.info("No bucket data to drill into.")
        return

    # Default to the largest current-period bucket so the user lands on
    # something meaningful, rather than the registry-first 'Brand'.
    buckets_sorted = (
        compare_df.sort_values("est_rev_curr", ascending=False)["campaign_bucket"].tolist()
    )
    bucket_choice = st.selectbox(
        "Campaign bucket",
        options=buckets_sorted,
        index=0,
        help="Sorted by current-period revenue.",
    )

    trend = daily_bucket_trend(
        paid_df, bucket_choice, period.curr_start, period.curr_end
    )
    if trend is None or trend.empty:
        st.info(f"No daily activity for bucket '{bucket_choice}' in the current window.")
        return

    trend_dt = pd.to_datetime(trend["day"])

    # ── Top row: marketing metrics (paid auction activity) ────────────
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Impressions / Clicks",
            "Cost (daily)",
            "Sessions",
            "CPC (daily)",
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=trend_dt, y=trend["impressioncount"],
            name="Impressions", marker_color=_PALETTE[0], opacity=0.85,
            hovertemplate="%{x|%b %-d}<br>Impressions: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trend_dt, y=trend["clickcount"], mode="lines+markers",
            name="Clicks", line=dict(color=_PALETTE[1], width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%b %-d}<br>Clicks: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1, secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=trend_dt, y=trend["cost"],
            marker_color=_PALETTE[3],
            hovertemplate="%{x|%b %-d}<br>Cost: $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=trend_dt, y=trend["sessions"],
            marker_color=_PALETTE[2],
            hovertemplate="%{x|%b %-d}<br>Sessions: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trend_dt, y=trend["cpc"], mode="lines+markers",
            line=dict(color=_PALETTE[4], width=2), marker=dict(size=5),
            hovertemplate="%{x|%b %-d}<br>CPC: $%{y:.2f}<extra></extra>",
        ),
        row=2, col=2,
    )
    fig.update_yaxes(title_text="Impressions", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Clicks", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Cost ($)", tickformat="$,.0f", row=1, col=2)
    fig.update_yaxes(title_text="Sessions", row=2, col=1)
    fig.update_yaxes(title_text="CPC ($)", tickformat="$,.2f", row=2, col=2)
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(tickformat="%b %-d", row=r, col=c, showgrid=False)

    fig.update_layout(
        height=600, margin=dict(l=60, r=40, t=60, b=50),
        title_text=f"Daily Trend — {bucket_choice} · {period.curr_start} → {period.curr_end}",
        title_font=dict(size=16),
        showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Cart + phone funnel summary (sums across current window) ──────
    curr_totals = aggregate_total_period(
        paid_df,
        period.curr_start, period.curr_end,
        buckets=[bucket_choice],
    )
    phone = bucket_phone_funnel(
        paid_df, bucket_choice, period.curr_start, period.curr_end
    )

    st.subheader(f"Funnel — {bucket_choice}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cart funnel**")
        cart_rows = [
            ("Sessions", _fmt_int(curr_totals["sessions"])),
            ("Cart Starts", _fmt_int(curr_totals["cart_starts"])),
            ("Cart Orders", _fmt_int(curr_totals["cart_orders2"])),
            ("Cart RR", _fmt_pct_value(curr_totals["cart_rr"])),
            ("Cart Conversion", _fmt_pct_value(curr_totals["cart_conversion"])),
            ("Cart VC", _fmt_pct_value(curr_totals["cart_vc"])),
        ]
        cart_df = pd.DataFrame(cart_rows, columns=["Metric", "Value"])
        st.dataframe(cart_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Phone funnel** (SERP vs Site split)")
        # Site queue calls = queue_calls - queue_serp (from the SQL).
        site_queue = phone["site_queue_calls"]
        site_phone = phone["site_phone_orders"]
        queue_other = phone["queue_calls_other"]
        phone_rows = [
            ("Gross Calls (total)", _fmt_int(phone["gross_calls"])),
            ("  • SERP", _fmt_int(phone["gross_serp"])),
            ("  • Site", _fmt_int(phone["gross_calls"] - phone["gross_serp"])),
            ("Queue Calls (total)", _fmt_int(phone["queue_calls"])),
            ("  • Grid", _fmt_int(phone["queue_calls_grid"])),
            ("  • Homepage", _fmt_int(phone["queue_calls_homepage"])),
            ("  • Other / non-SERP", _fmt_int(queue_other)),
            ("  • SERP", _fmt_int(phone["queue_serp"])),
            ("Net Calls", _fmt_int(phone["net_calls"])),
            ("Phone Orders (total)", _fmt_int(phone["phone_orders"])),
            ("  • Site (non-SERP)", _fmt_int(site_phone)),
            ("  • SERP", _fmt_int(phone["serp_orders"])),
            ("Site RR", _fmt_pct_value(
                site_queue / curr_totals["sessions"]
                if curr_totals["sessions"] else 0.0
            )),
            ("Site Conversion", _fmt_pct_value(
                site_phone / site_queue if site_queue else 0.0
            )),
        ]
        phone_df = pd.DataFrame(phone_rows, columns=["Metric", "Value"])
        st.dataframe(phone_df, use_container_width=True, hide_index=True)

    # ── Efficiency metric cards ───────────────────────────────────────
    st.subheader("Efficiency")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("CPC", _fmt_money_precise(curr_totals["cpc"]))
    e2.metric("Cost / Order", _fmt_money_precise(curr_totals["cost_per_order"]))
    e3.metric("Revenue / Session", _fmt_money_precise(curr_totals["revenue_per_session"]))
    e4.metric("ROAS", _fmt_float(curr_totals["roas"], 2))

    # ── Efficiency daily trend ────────────────────────────────────────
    eff_fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Cost / Order", "Revenue / Session", "ROAS"),
        horizontal_spacing=0.12,
    )
    eff_fig.add_trace(
        go.Scatter(
            x=trend_dt, y=trend["cost_per_order"], mode="lines+markers",
            line=dict(color=_PALETTE[5], width=2), marker=dict(size=5),
            hovertemplate="%{x|%b %-d}<br>CPO: $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    eff_fig.add_trace(
        go.Scatter(
            x=trend_dt, y=trend["revenue_per_session"], mode="lines+markers",
            line=dict(color=_PALETTE[6], width=2), marker=dict(size=5),
            hovertemplate="%{x|%b %-d}<br>Rev/Session: $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=2,
    )
    eff_fig.add_trace(
        go.Scatter(
            x=trend_dt, y=trend["roas"], mode="lines+markers",
            line=dict(color=_PALETTE[7], width=2), marker=dict(size=5),
            hovertemplate="%{x|%b %-d}<br>ROAS: %{y:,.2f}<extra></extra>",
        ),
        row=1, col=3,
    )
    eff_fig.update_yaxes(title_text="$ / Order", tickformat="$,.0f", row=1, col=1)
    eff_fig.update_yaxes(title_text="$ / Session", tickformat="$,.2f", row=1, col=2)
    eff_fig.update_yaxes(title_text="ROAS", row=1, col=3)
    for c in (1, 2, 3):
        eff_fig.update_xaxes(tickformat="%b %-d", row=1, col=c, showgrid=False)
    eff_fig.update_layout(
        height=340, margin=dict(l=60, r=40, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    st.plotly_chart(eff_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(ctx: AppContext) -> None:
    st.title("Deep Dive — Paid Search")
    _render_sidebar_note()

    # Tab-local paid window picker (drives Sections 3–5).
    st.subheader("Paid comparison window")
    period = _render_period_picker()

    # ── Sections 1–2 (sidebar-driven finance) ─────────────────────────
    st.divider()
    _render_pacing_summary(ctx)
    st.divider()
    _render_waterfall(ctx)
    st.divider()

    # ── Sections 3–5 (paid-window-driven) ─────────────────────────────
    # One Databricks round-trip spans both windows; the three sections
    # slice off that single frame in-memory.
    try:
        with st.spinner("Loading Paid Search campaign data…"):
            paid_df = fetch_paid_for_windows(
                period.curr_start, period.curr_end,
                period.prior_start, period.prior_end,
            )
    except Exception as e:  # pragma: no cover — degrade gracefully
        st.error(f"Paid Search data unavailable: {e}")
        return

    if paid_df is None or paid_df.empty:
        st.info("No paid search campaign rows returned for the selected windows.")
        return

    ctx.cache["paid_bucket_df"] = paid_df
    ctx.cache["paid_period"] = period

    compare_df = pd.DataFrame()
    try:
        compare_df = _render_bucket_performance(period, paid_df)
        ctx.cache["paid_bucket_compare"] = compare_df
    except Exception as e:
        st.warning(f"Campaign bucket performance unavailable: {e}")
    st.divider()

    try:
        _render_bucket_vc_waterfall(period, paid_df)
    except Exception as e:
        st.warning(f"Bucket VC decomposition unavailable: {e}")
    st.divider()

    try:
        _render_bucket_drilldown(period, paid_df, compare_df)
    except Exception as e:
        st.warning(f"Bucket drill-down unavailable: {e}")
