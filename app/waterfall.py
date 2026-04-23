"""
Revenue Waterfall — Pacing vs Plan decomposition.

Implements the sequential-substitution method from the PRD (and the SEO
reporting agent rule): swap each driver from Plan → Actual one at a time,
left to right, attributing the resulting revenue delta to that driver.

    Phone Revenue = Sessions x Site RR x Site Conversion x Phone GCV/Order
    Cart Revenue  = Sessions x Cart RR x Cart Conversion x Cart GCV/Order
    Total Revenue = Phone Revenue + Cart Revenue

Seven drivers, evaluated in this order:
    1. Sessions          (affects both paths)
    2. Site RR           (phone path only)
    3. Site Conversion   (phone path only)
    4. Phone GCV/Order   (phone path only)
    5. Cart RR           (cart path only)
    6. Cart Conversion   (cart path only)
    7. Cart GCV/Order    (cart path only)

By construction the seven impacts sum exactly to `Pacing Revenue - Plan Revenue`.

Input: the pacing DataFrame produced by `finance_data.fetch_plan_pacing`,
filtered to the selected UI channels. The pacing DataFrame gives us raw
aggregates (sessions, orders, revenue); this module reconstructs the rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.finance_data import _map_channels_for_plan


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WaterfallDrivers:
    """Extracted per-path driver values for a single scenario (Plan or Pacing)."""

    sessions: float
    site_rr: float          # site_queue_calls / sessions
    site_conversion: float  # site_phone_orders / site_queue_calls
    phone_gcv_order: float  # phone_revenue / site_phone_orders
    cart_rr: float          # cart_entries / sessions
    cart_conversion: float  # cart_orders / cart_entries
    cart_gcv_order: float   # cart_revenue / cart_orders

    @property
    def phone_revenue(self) -> float:
        return self.sessions * self.site_rr * self.site_conversion * self.phone_gcv_order

    @property
    def cart_revenue(self) -> float:
        return self.sessions * self.cart_rr * self.cart_conversion * self.cart_gcv_order

    @property
    def total_revenue(self) -> float:
        return self.phone_revenue + self.cart_revenue


@dataclass
class WaterfallResult:
    """Output of the sequential-substitution waterfall."""

    plan: WaterfallDrivers
    actual: WaterfallDrivers

    plan_revenue: float
    pacing_revenue: float

    sessions_impact: float
    site_rr_impact: float
    site_conversion_impact: float
    phone_gcv_impact: float
    cart_rr_impact: float
    cart_conversion_impact: float
    cart_gcv_impact: float

    @property
    def total_gap(self) -> float:
        return self.pacing_revenue - self.plan_revenue

    @property
    def impacts(self) -> list[tuple[str, float]]:
        """Ordered (label, dollar_impact) pairs, matching the substitution order."""
        return [
            ("Sessions", self.sessions_impact),
            ("Site RR", self.site_rr_impact),
            ("Site Conversion", self.site_conversion_impact),
            ("Phone GCV/Order", self.phone_gcv_impact),
            ("Cart RR", self.cart_rr_impact),
            ("Cart Conversion", self.cart_conversion_impact),
            ("Cart GCV/Order", self.cart_gcv_impact),
        ]


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """Return num/den, 0 when denominator is zero/missing."""
    if not den:
        return 0.0
    return num / den


def _extract_drivers(agg: dict) -> WaterfallDrivers:
    """Reconstruct the seven drivers from aggregated pacing columns."""
    sessions = float(agg.get("sessions", 0) or 0)
    site_queue_calls = float(agg.get("site_queue_calls", 0) or 0)
    site_phone_orders = float(agg.get("site_phone_orders", 0) or 0)
    phone_revenue = float(agg.get("phone_revenue", 0) or 0)
    cart_entries = float(agg.get("cart_entries", 0) or 0)
    cart_orders = float(agg.get("cart_orders", 0) or 0)
    cart_revenue = float(agg.get("cart_revenue", 0) or 0)

    return WaterfallDrivers(
        sessions=sessions,
        site_rr=_safe_div(site_queue_calls, sessions),
        site_conversion=_safe_div(site_phone_orders, site_queue_calls),
        phone_gcv_order=_safe_div(phone_revenue, site_phone_orders),
        cart_rr=_safe_div(cart_entries, sessions),
        cart_conversion=_safe_div(cart_orders, cart_entries),
        cart_gcv_order=_safe_div(cart_revenue, cart_orders),
    )


def _aggregate(df: pd.DataFrame) -> dict:
    cols = [
        "sessions", "site_queue_calls", "site_phone_orders", "phone_revenue",
        "cart_entries", "cart_orders", "cart_revenue",
    ]
    return {c: df[c].sum() if c in df.columns else 0.0 for c in cols}


def compute_revenue_waterfall(
    plan_df: pd.DataFrame,
    channels: list[str] | None = None,
) -> WaterfallResult | None:
    """
    Run the sequential-substitution waterfall on the pacing DataFrame.

    Parameters
    ----------
    plan_df : DataFrame
        Output of `fetch_plan_pacing()`. Must contain `perf_view` ('Pacing'/'Plan'),
        `MarketingChannel`, and the aggregate columns.
    channels : list[str] | None
        UI-level channel names to filter to. None / empty means all channels.

    Returns
    -------
    WaterfallResult or None
        None if either Plan or Pacing rows are missing after filtering, or
        if Plan revenue is zero (which would make the decomposition meaningless).
    """
    if plan_df is None or plan_df.empty:
        return None

    df = plan_df
    if channels:
        mapped = _map_channels_for_plan(channels)
        df = df[df["MarketingChannel"].isin(mapped)]

    if df.empty:
        return None

    # `fetch_plan_pacing` returns one row per past-month-end snapshot plus a
    # mid-month snapshot for the in-flight month. We want the waterfall to
    # describe a single month — use the latest available `rpt_date`, which is
    # either the current MTD (mid-month) or the most recent locked month-end.
    if "rpt_date" in df.columns and not df["rpt_date"].isna().all():
        latest = df["rpt_date"].max()
        df = df[df["rpt_date"] == latest]

    pac = df[df["perf_view"] == "Pacing"]
    pln = df[df["perf_view"] == "Plan"]
    if pac.empty or pln.empty:
        return None

    P = _extract_drivers(_aggregate(pln))
    A = _extract_drivers(_aggregate(pac))

    plan_phone = P.sessions * P.site_rr * P.site_conversion * P.phone_gcv_order
    plan_cart  = P.sessions * P.cart_rr * P.cart_conversion * P.cart_gcv_order
    plan_rev   = plan_phone + plan_cart

    # Step 1 — Sessions (affects both paths simultaneously)
    after_sess_phone = A.sessions * P.site_rr * P.site_conversion * P.phone_gcv_order
    after_sess_cart  = A.sessions * P.cart_rr * P.cart_conversion * P.cart_gcv_order
    sessions_impact  = (after_sess_phone + after_sess_cart) - plan_rev

    # Step 2 — Site RR (phone path only)
    after_srr_phone = A.sessions * A.site_rr * P.site_conversion * P.phone_gcv_order
    srr_impact      = after_srr_phone - after_sess_phone

    # Step 3 — Site Conversion
    after_sconv_phone = A.sessions * A.site_rr * A.site_conversion * P.phone_gcv_order
    sconv_impact      = after_sconv_phone - after_srr_phone

    # Step 4 — Phone GCV/Order
    actual_phone  = A.sessions * A.site_rr * A.site_conversion * A.phone_gcv_order
    pgcv_impact   = actual_phone - after_sconv_phone

    # Step 5 — Cart RR (cart path only)
    after_crr_cart = A.sessions * A.cart_rr * P.cart_conversion * P.cart_gcv_order
    crr_impact     = after_crr_cart - after_sess_cart

    # Step 6 — Cart Conversion
    after_cconv_cart = A.sessions * A.cart_rr * A.cart_conversion * P.cart_gcv_order
    cconv_impact     = after_cconv_cart - after_crr_cart

    # Step 7 — Cart GCV/Order
    actual_cart  = A.sessions * A.cart_rr * A.cart_conversion * A.cart_gcv_order
    cgcv_impact  = actual_cart - after_cconv_cart

    pacing_rev = actual_phone + actual_cart

    return WaterfallResult(
        plan=P,
        actual=A,
        plan_revenue=plan_rev,
        pacing_revenue=pacing_rev,
        sessions_impact=sessions_impact,
        site_rr_impact=srr_impact,
        site_conversion_impact=sconv_impact,
        phone_gcv_impact=pgcv_impact,
        cart_rr_impact=crr_impact,
        cart_conversion_impact=cconv_impact,
        cart_gcv_impact=cgcv_impact,
    )


# ---------------------------------------------------------------------------
# Chart + narrative
# ---------------------------------------------------------------------------

_BAR_TOTAL = "rgb(44, 62, 80)"
_BAR_UP    = "rgb(39, 174, 96)"
_BAR_DOWN  = "rgb(231, 76, 60)"


def build_waterfall_figure(result: WaterfallResult, title: str | None = None) -> go.Figure:
    """Plotly waterfall chart: Plan Revenue → 7 driver bars → Pacing Revenue.

    The y-axis is zoomed to start below the smaller of the two totals so the
    seven driver bars (which are much smaller than Plan / Pacing Revenue in
    absolute terms) aren't visually flattened. This makes the chart slide-ready
    without changing the underlying math.
    """
    impacts = result.impacts
    labels  = ["Plan Revenue"] + [name for name, _ in impacts] + ["Pacing Revenue"]
    values  = [result.plan_revenue] + [v for _, v in impacts] + [result.pacing_revenue]
    measure = ["absolute"] + ["relative"] * len(impacts) + ["total"]

    gap = result.total_gap
    text: list[str] = []
    for m, v in zip(measure, values):
        if m in ("absolute", "total"):
            text.append(f"<b>${v:,.0f}</b>")
            continue
        sign = "+" if v >= 0 else "−"  # Unicode minus renders cleaner than ASCII
        amt  = f"{sign}${abs(v):,.0f}"
        pct  = f"<br>({v / gap * 100:+.0f}% of gap)" if gap else ""
        text.append(f"<b>{amt}</b>{pct}")

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measure,
        text=text,
        textposition="outside",
        textfont=dict(size=14, color="#1a1a1a"),
        connector=dict(line=dict(color="rgb(160,160,160)", width=1, dash="dot")),
        increasing=dict(marker=dict(color=_BAR_UP,   line=dict(color="#145a32", width=1))),
        decreasing=dict(marker=dict(color=_BAR_DOWN, line=dict(color="#7b241c", width=1))),
        totals=dict(marker=dict(color=_BAR_TOTAL, line=dict(color="#17202a", width=1))),
        width=0.6,
    ))

    # --- Zoom y-axis so the driver bars aren't dwarfed by the totals ---
    # Base = smaller of the two totals. We start the axis a bit below that so
    # the shorter total still has a visible "stub", then give plenty of
    # headroom above so the outside labels don't clip.
    total_min = min(result.plan_revenue, result.pacing_revenue)
    total_max = max(result.plan_revenue, result.pacing_revenue)
    # Vertical span within the driver region (totals + per-driver swings)
    driver_span = max(
        total_max - total_min,
        max(abs(v) for _, v in impacts) if impacts else 0.0,
    )
    y_floor   = max(0.0, total_min - driver_span * 0.35)
    y_ceiling = total_max + driver_span * 0.55  # headroom for outside text

    fig.update_layout(
        title=dict(
            text=title or "Revenue Waterfall — Plan → Pacing",
            font=dict(size=18, color="#1a1a1a"),
            x=0.02,
            xanchor="left",
        ),
        yaxis=dict(
            title=dict(text="Revenue ($)", font=dict(size=14)),
            tickformat="$,.0f",
            tickfont=dict(size=13),
            range=[y_floor, y_ceiling],
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=15, color="#1a1a1a", family="Arial Black, Arial, sans-serif"),
            ticks="outside",
            ticklen=6,
            showgrid=False,
        ),
        margin=dict(l=80, r=40, t=80, b=130),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=560,
        uniformtext=dict(minsize=12, mode="show"),
    )
    # Subtle horizontal baseline at Plan Revenue so vs-Plan magnitude reads fast.
    fig.add_hline(
        y=result.plan_revenue,
        line=dict(color="rgba(44,62,80,0.35)", width=1, dash="dash"),
    )
    return fig


def build_narrative(result: WaterfallResult) -> str:
    """One-line callout highlighting the biggest positive and negative impacts.

    Every ``$`` is escaped as ``\\$`` so Streamlit's markdown renderer does not
    interpret the dollar amounts as KaTeX math delimiters.
    """
    impacts = result.impacts
    negatives = [(n, v) for n, v in impacts if v < 0]
    positives = [(n, v) for n, v in impacts if v > 0]

    def money(v: float) -> str:
        # Escape `$` so Streamlit doesn't treat it as a math fence, and keep
        # the sign outside the dollar amount (e.g. `-$1,234` not `$-1,234`).
        sign = "-" if v < 0 else ""
        return f"{sign}\\${abs(v):,.0f}"

    pieces: list[str] = []
    gap = result.total_gap
    if gap >= 0:
        pieces.append(
            f"Pacing is **{money(result.pacing_revenue)}** vs Plan "
            f"**{money(result.plan_revenue)}** — ahead by **{money(gap)}**."
        )
    else:
        pieces.append(
            f"Pacing is **{money(result.pacing_revenue)}** vs Plan "
            f"**{money(result.plan_revenue)}** — short by **{money(-gap)}**."
        )

    if negatives:
        worst = min(negatives, key=lambda x: x[1])
        pieces.append(f"Biggest drag: **{worst[0]} ({money(worst[1])})**.")
    if positives:
        best = max(positives, key=lambda x: x[1])
        pieces.append(f"Biggest tailwind: **{best[0]} (+{money(best[1])})**.")

    return " ".join(pieces)


def build_impact_table(result: WaterfallResult) -> pd.DataFrame:
    """Per-driver Plan/Actual values + dollar impact + % of gap."""
    plan = result.plan
    actual = result.actual

    def pct(val: float, base: float) -> str:
        if not base:
            return "—"
        return f"{(val / base - 1) * 100:+.1f}%"

    def fmt_rate(v: float) -> str:
        return f"{v * 100:.2f}%"

    def fmt_num(v: float) -> str:
        return f"{v:,.0f}"

    def fmt_money(v: float) -> str:
        return f"${v:,.2f}"

    rows = [
        ("Sessions",         fmt_num(plan.sessions),          fmt_num(actual.sessions),
         pct(actual.sessions, plan.sessions),                  result.sessions_impact),
        ("Site RR",          fmt_rate(plan.site_rr),          fmt_rate(actual.site_rr),
         pct(actual.site_rr, plan.site_rr),                    result.site_rr_impact),
        ("Site Conversion",  fmt_rate(plan.site_conversion),  fmt_rate(actual.site_conversion),
         pct(actual.site_conversion, plan.site_conversion),    result.site_conversion_impact),
        ("Phone GCV/Order",  fmt_money(plan.phone_gcv_order), fmt_money(actual.phone_gcv_order),
         pct(actual.phone_gcv_order, plan.phone_gcv_order),    result.phone_gcv_impact),
        ("Cart RR",          fmt_rate(plan.cart_rr),          fmt_rate(actual.cart_rr),
         pct(actual.cart_rr, plan.cart_rr),                    result.cart_rr_impact),
        ("Cart Conversion",  fmt_rate(plan.cart_conversion),  fmt_rate(actual.cart_conversion),
         pct(actual.cart_conversion, plan.cart_conversion),    result.cart_conversion_impact),
        ("Cart GCV/Order",   fmt_money(plan.cart_gcv_order),  fmt_money(actual.cart_gcv_order),
         pct(actual.cart_gcv_order, plan.cart_gcv_order),      result.cart_gcv_impact),
    ]

    gap = result.total_gap or 1.0
    return pd.DataFrame([
        {
            "Driver": name,
            "Plan": plan_val,
            "Pacing": pacing_val,
            "vs Plan": vp,
            "$ Impact": f"{'+' if dollar >= 0 else ''}${dollar:,.0f}",
            "% of Gap": f"{dollar / gap * 100:+.0f}%" if result.total_gap else "—",
        }
        for name, plan_val, pacing_val, vp, dollar in rows
    ])


# ---------------------------------------------------------------------------
# Streamlit section renderer
# ---------------------------------------------------------------------------

def render_waterfall_section(
    plan_df: pd.DataFrame | None,
    channels: list[str] | None,
    *,
    chart_title: str = "Revenue Waterfall — Plan → Pacing",
    caption: str | None = None,
) -> WaterfallResult | None:
    """
    Render the full Pacing-vs-Plan revenue waterfall section.

    Wraps the compute + chart + narrative + detail table so any tab can call
    this in a single line. Returns the computed `WaterfallResult` (or None if
    data was insufficient) so callers can feed it downstream (e.g. LLM narrative).
    """
    if plan_df is None:
        st.warning("Revenue waterfall unavailable — plan/pacing data did not load.")
        return None

    result = compute_revenue_waterfall(plan_df, channels)
    if result is None:
        st.info(
            "Not enough data to build the waterfall for the selected channels. "
            "Need both Pacing and Plan rows from `rpt_texas_daily_pacing`."
        )
        return None
    if result.plan_revenue <= 0:
        st.info(
            "Plan revenue is zero for the selected channels — waterfall would be "
            "degenerate. Widen the channel selection or wait for plan data to load."
        )
        return None

    st.plotly_chart(build_waterfall_figure(result, title=chart_title), use_container_width=True)
    st.markdown(build_narrative(result))
    if caption:
        st.caption(caption)

    # Residual check — sequential substitution should reconcile to $0.
    reconciled = sum(v for _, v in result.impacts)
    residual = result.total_gap - reconciled
    if abs(residual) > 1.0:
        st.caption(
            f"⚠️ Waterfall residual of \\${residual:,.2f} — driver impacts do not "
            "fully reconcile to the Plan→Pacing gap. Check upstream data."
        )

    with st.expander("Driver detail (Plan vs Pacing, per-driver \\$ impact)"):
        st.dataframe(build_impact_table(result), use_container_width=True, hide_index=True)
        st.caption(
            "Method: sequential substitution. Each row swaps one driver from "
            "Plan → Actual while holding prior drivers at Actual and subsequent "
            "drivers at Plan, so the seven impacts sum exactly to Pacing − Plan."
        )

    return result
