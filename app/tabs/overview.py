"""
Overview tab (all channels) — the primary v2 view.

Renders the original Sections 1-7 from v1 plus a stub for the new Revenue
Waterfall (Section 4) that Phase 2 will fill in. The "Ask the Analyst" chat
was lifted out of this tab into its own tab in v2.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.app_context import AppContext
from app.config import DIMENSION_DISPLAY_NAMES, DRIVER_DIMENSIONS, KPIS
from app.decomposition import (
    _initiative_label,
    analyze_initiatives,
    compute_initiative_impact,
    decompose_all_dimensions,
    rank_top_drivers,
)
from app.finance_data import build_funnel_summary, render_summary_html
from app.kpi_engine import compute_funnel_table, compute_kpi_summary
from app.narrative import generate_llm_narrative
from app.waterfall import render_waterfall_section


def _fmt_pp(delta: float) -> str:
    return f"{delta * 100:+.2f}pp"


def _fmt_pct_change(delta: float, base: float) -> str:
    if base == 0:
        return "N/A"
    return f"{(delta / base) * 100:+.1f}%"


def _dim_label(dim: str) -> str:
    return DIMENSION_DISPLAY_NAMES.get(dim, dim.replace("_", " ").title())


def _render_performance_summary(ctx: AppContext) -> None:
    """Section 1 — Finance source-of-truth funnel summary."""
    if ctx.finance_df is None or ctx.plan_df is None:
        st.warning("Finance performance summary unavailable (data fetch failed).")
        return

    try:
        rows = build_funnel_summary(
            ctx.finance_df,
            ctx.plan_df,
            ctx.effective_channels,
            curr_start=ctx.curr_start,
            curr_end=ctx.curr_end,
            prior_start=ctx.prior_start,
            prior_end=ctx.prior_end,
        )

        st.header("Performance Summary")
        st.caption(
            "Source: **Finance reporting queries** (source of truth for performance). "
            "Session-level driver analysis below is directional and may not tie exactly."
        )
        st.markdown(
            render_summary_html(rows, date.today(), period_label=ctx.period_label),
            unsafe_allow_html=True,
        )
        st.divider()
    except Exception as e:  # pragma: no cover - defensive
        st.warning(f"Finance summary unavailable: {e}")


def _render_kpi_summary(ctx: AppContext) -> dict:
    """Section 2 — Selected KPI metric cards + full funnel expander."""
    kpi_key = ctx.kpi_key
    st.header(f"1. {KPIS[kpi_key].name} — Period Summary")
    st.caption(
        "Source: **Session-level data** (Tier 2) — directional for decomposition. "
        "For official performance numbers see the Finance summary above."
    )

    summary = compute_kpi_summary(ctx.df_current, ctx.df_prior, kpi_key)
    pct_change_str = _fmt_pct_change(summary["delta"], summary["prior_rate"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Current",
        f"{summary['current_rate'] * 100:.2f}%",
        delta=f"{pct_change_str} ({_fmt_pp(summary['delta'])})",
    )
    c2.metric("Prior", f"{summary['prior_rate'] * 100:.2f}%")
    c3.metric(
        "Current volume",
        f"{int(summary['current_numerator']):,} / {int(summary['current_denominator']):,}",
    )
    c4.metric(
        "Prior volume",
        f"{int(summary['prior_numerator']):,} / {int(summary['prior_denominator']):,}",
    )

    with st.expander("Full funnel summary", expanded=False):
        funnel_df = compute_funnel_table(ctx.df_current, ctx.df_prior)
        display_df = funnel_df.copy()
        for c in ["current_rate", "prior_rate"]:
            display_df[c] = (display_df[c] * 100).round(2).astype(str) + "%"
        display_df["delta"] = (funnel_df["delta"] * 100).map(lambda x: f"{x:+.2f}pp")
        display_df["pct_change"] = (
            (funnel_df["pct_change"] * 100).round(1).astype(str) + "%"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    return summary


def _render_drivers(ctx: AppContext, summary: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Section 3 — What Drove the Change (drivers + initiative WoW bars)."""
    st.header("2. What Drove the Change")
    st.caption(
        "Source: **Session-level decomposition** (Tier 2). Mix/rate effects explain "
        "*why* the KPI moved but may not tie exactly to finance totals."
    )

    kpi_key = ctx.kpi_key

    # Landing page type only meaningful when restricted to Organic.
    organic_only = bool(ctx.channel_filter) and all(
        c.lower() == "organic" for c in ctx.channel_filter
    )
    driver_dims = [
        d for d in DRIVER_DIMENSIONS if d != "landing_page_type" or organic_only
    ]

    decomp_results = decompose_all_dimensions(
        ctx.df_current, ctx.df_prior, kpi_key, driver_dims
    )
    top_drivers = rank_top_drivers(decomp_results, n=8)

    impact_current = compute_initiative_impact(ctx.df_current, kpi_key)
    impact_prior = compute_initiative_impact(ctx.df_prior, kpi_key)
    init_current = analyze_initiatives(ctx.df_current, kpi_key)

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

    if not impact_current.empty and all_bars:
        min_dim_bar = min(abs(b["pp"]) for b in all_bars)
        init_delta = pd.merge(
            impact_current[["initiative", "scaled_impact_on_kpi"]].rename(
                columns={"scaled_impact_on_kpi": "curr_impact"}
            ),
            impact_prior[["initiative", "scaled_impact_on_kpi"]].rename(
                columns={"scaled_impact_on_kpi": "prior_impact"}
            ) if not impact_prior.empty else pd.DataFrame(columns=["initiative", "prior_impact"]),
            on="initiative", how="left",
        ).fillna(0)
        init_delta["wow_change"] = init_delta["curr_impact"] - init_delta["prior_impact"]

        for _, row in init_delta.iterrows():
            wow_pp = row["wow_change"] * 100
            if abs(wow_pp) >= min_dim_bar:
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
        pct_change_str = _fmt_pct_change(summary["delta"], summary["prior_rate"])
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

    return top_drivers, init_current, impact_current, impact_prior


def _render_revenue_waterfall(ctx: AppContext) -> None:
    """Section 4 — Pacing vs Plan revenue waterfall (sequential substitution)."""
    st.header("3. Revenue Waterfall")
    st.caption(
        "Pacing vs Plan revenue gap decomposed into per-driver dollar impacts. "
        "Drivers are substituted Plan → Actual sequentially (Sessions → Site RR "
        "→ Site Conversion → Phone GCV/Order → Cart RR → Cart Conversion → "
        "Cart GCV/Order), so the seven bars sum exactly to the total gap."
    )

    channels = ctx.effective_channels
    channel_label = (
        ", ".join(channels) if channels and len(channels) <= 4
        else f"{len(channels)} channels" if channels else "All channels"
    )
    result = render_waterfall_section(
        ctx.plan_df,
        channels,
        chart_title=f"Revenue Waterfall — Plan → Pacing · {channel_label}",
        caption="Source: `energy_prod.energy.rpt_texas_daily_pacing` (Pacing + Final/Plan views).",
    )

    # Cache so the analyst-chat tab can reuse without recomputing.
    if result is not None:
        ctx.cache["overview_waterfall"] = result


def _render_dimension_detail(ctx: AppContext) -> None:
    """Section 5 — per-dimension drill-down tables."""
    st.header("4. Dimension Detail")

    all_decomp = decompose_all_dimensions(
        ctx.df_current, ctx.df_prior, ctx.kpi_key, DRIVER_DIMENSIONS
    )
    visible_dims = [d for d in DRIVER_DIMENSIONS if d in all_decomp]
    if not visible_dims:
        st.info("No dimension data available for this selection.")
        return

    dim_tabs = st.tabs([_dim_label(d) for d in visible_dims])
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
    for tab, dim in zip(dim_tabs, visible_dims):
        with tab:
            detail = all_decomp[dim].copy()
            for c in ["curr_rate", "prior_rate"]:
                if c in detail.columns:
                    detail[c] = (detail[c] * 100).round(2).astype(str) + "%"
            for c in ["curr_weight", "prior_weight"]:
                if c in detail.columns:
                    detail[c] = (detail[c] * 100).round(1).astype(str) + "%"
            for c in ["mix_effect", "rate_effect", "total_contribution"]:
                if c in detail.columns:
                    detail[c] = (detail[c] * 100).map(lambda x: f"{x:+.2f}pp")
            detail = detail.rename(columns={k: v for k, v in col_rename.items() if k in detail.columns})
            st.dataframe(detail, use_container_width=True, hide_index=True)


def _render_initiative_impact(
    ctx: AppContext,
    impact_current: pd.DataFrame,
    impact_prior: pd.DataFrame,
) -> None:
    """Section 6 — Initiative Impact (model vs holdout)."""
    st.header("5. Initiative Impact")
    st.caption(
        "Source: **Session-level initiative flags** (Tier 2). "
        "Each initiative's lift (model rate − holdout rate) is scaled by its "
        "share of traffic to show the actual contribution to the all-in KPI."
    )

    if impact_current.empty:
        st.info("No initiative data available for this period.")
        return

    total_curr = impact_current["scaled_impact_on_kpi"].sum()
    total_prior = (
        impact_prior["scaled_impact_on_kpi"].sum() if not impact_prior.empty else 0.0
    )
    change = total_curr - total_prior

    ic1, ic2, ic3 = st.columns(3)
    ic1.metric(
        "Total initiative contribution",
        _fmt_pp(total_curr),
        help="Sum of all model/FMP lifts scaled by traffic share",
    )
    ic2.metric("Prior period contribution", _fmt_pp(total_prior))
    ic3.metric("Change in initiative contribution", _fmt_pp(change), delta=_fmt_pp(change))

    chart_impact = impact_current.sort_values("scaled_impact_on_kpi", ascending=True)
    fig = go.Figure(go.Bar(
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
    fig.update_layout(
        title=f"Initiative contribution to {KPIS[ctx.kpi_key].short_name} (current period)",
        xaxis_title="Scaled impact on all-in KPI (pp)",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="gray"),
        yaxis_title="",
        height=max(300, len(chart_impact) * 60),
        margin=dict(l=20, r=100),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Period-over-period comparison")
    merged = pd.merge(
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

    merged["lift_change"] = merged["curr_lift"] - merged["prior_lift"]
    merged["impact_change"] = merged["curr_impact"] - merged["prior_impact"]

    display = pd.DataFrame({
        "Initiative": merged["initiative"],
        "Curr Model Rate": (merged["curr_model_rate"] * 100).round(2).astype(str) + "%",
        "Curr Holdout Rate": (merged["curr_holdout_rate"] * 100).round(2).astype(str) + "%",
        "Curr Lift": merged["curr_lift"].map(lambda x: f"{x * 100:+.2f}pp"),
        "Prior Lift": merged["prior_lift"].map(lambda x: f"{x * 100:+.2f}pp"),
        "Lift Change": merged["lift_change"].map(lambda x: f"{x * 100:+.2f}pp"),
        "Curr Share": (merged["curr_share"] * 100).round(1).astype(str) + "%",
        "Curr Impact": merged["curr_impact"].map(lambda x: f"{x * 100:+.3f}pp"),
        "Prior Impact": merged["prior_impact"].map(lambda x: f"{x * 100:+.3f}pp"),
        "Impact Change": merged["impact_change"].map(lambda x: f"{x * 100:+.3f}pp"),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "**Lift** = model rate − holdout rate. "
        "**Share** = initiative's share of the KPI denominator. "
        "**Scaled Impact** = lift × share = how much the initiative adds to the all-in KPI vs holdout. "
        "**Impact Change** = did the initiative's contribution improve or worsen vs prior period?"
    )


def _render_cart_conversion_decomp(ctx: AppContext) -> None:
    """Section 7 — only for Cart Conversion / VC."""
    if ctx.kpi_key not in ("Cart Conversion", "VC"):
        return
    st.header("6. Cart Conversion Decomposition")
    st.caption(
        "Cart Conversion = SSN Submit Rate × Conversion After Credit. "
        "Isolates pre-credit friction from post-credit eligibility."
    )
    sub_kpis = ["SSN Submit Rate", "Conversion After Credit"]
    cols = st.columns(len(sub_kpis))
    for col, sk in zip(cols, sub_kpis):
        with col:
            s = compute_kpi_summary(ctx.df_current, ctx.df_prior, sk)
            pct_ch = _fmt_pct_change(s["delta"], s["prior_rate"])
            col.metric(
                KPIS[sk].name,
                f"{s['current_rate'] * 100:.2f}%",
                delta=f"{pct_ch} ({_fmt_pp(s['delta'])})",
            )


def _render_narrative(
    ctx: AppContext,
    summary: dict,
    top_drivers: pd.DataFrame,
    init_current: pd.DataFrame,
    impact_current: pd.DataFrame,
    impact_prior: pd.DataFrame,
) -> None:
    """Section 8 — LLM-generated weekly summary."""
    st.header("7. Weekly Review Summary")
    st.caption("AI-generated commentary — review before sharing.")

    with st.spinner("Generating summary with GPT-4o…"):
        narrative = generate_llm_narrative(
            summary, top_drivers, init_current,
            initiative_impact=impact_current if not impact_current.empty else None,
            initiative_impact_prior=impact_prior if not impact_prior.empty else None,
        )

    st.markdown(narrative)

    # Stash the computed artefacts so the Ask-the-Analyst tab can reuse them
    # without recomputing.
    ctx.cache["overview_summary"] = summary
    ctx.cache["overview_top_drivers"] = top_drivers
    ctx.cache["overview_init_current"] = init_current
    ctx.cache["overview_impact_current"] = impact_current
    ctx.cache["overview_impact_prior"] = impact_prior


def render(ctx: AppContext) -> None:
    """Render the Overview tab."""
    _render_performance_summary(ctx)
    summary = _render_kpi_summary(ctx)
    top_drivers, init_current, impact_current, impact_prior = _render_drivers(ctx, summary)
    _render_revenue_waterfall(ctx)
    _render_dimension_detail(ctx)
    _render_initiative_impact(ctx, impact_current, impact_prior)
    _render_cart_conversion_decomp(ctx)
    _render_narrative(ctx, summary, top_drivers, init_current, impact_current, impact_prior)
