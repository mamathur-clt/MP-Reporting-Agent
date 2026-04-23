"""
Chart rendering for the Slack SEO bot.

Generates styled Plotly chart images (PNG bytes) for:
1. Pacing Snapshot table -- full funnel with conditional formatting
2. Revenue Waterfall -- sequential substitution decomposition
3. Top-of-Funnel Performance table -- GSC metrics with deltas
"""

import io
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

_GREEN_BG = "rgba(0, 150, 0, 0.12)"
_RED_BG = "rgba(200, 0, 0, 0.12)"
_NEUTRAL_BG = "white"

_GREEN_TEXT = "rgb(0, 130, 0)"
_RED_TEXT = "rgb(190, 0, 0)"
_NEUTRAL_TEXT = "rgb(80, 80, 80)"

_HEADER_BG = "rgb(44, 62, 80)"
_HEADER_TEXT = "white"
_ALT_ROW_BG = "rgba(240, 240, 240, 0.5)"

_FONT_FAMILY = "Arial, Helvetica, sans-serif"

_THRESHOLD = 0.025  # 2.5%


def _delta_bg(val: float) -> str:
    if pd.isna(val):
        return _NEUTRAL_BG
    if val > _THRESHOLD:
        return _GREEN_BG
    if val < -_THRESHOLD:
        return _RED_BG
    return _NEUTRAL_BG


def _delta_text_color(val: float) -> str:
    if pd.isna(val):
        return _NEUTRAL_TEXT
    if val > _THRESHOLD:
        return _GREEN_TEXT
    if val < -_THRESHOLD:
        return _RED_TEXT
    return _NEUTRAL_TEXT


def _fmt_pct(val: float) -> str:
    """Format a fractional delta as a signed percentage string."""
    if pd.isna(val):
        return "—"
    pct = val * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _fmt_rate(val: float) -> str:
    """Format a rate (already decimal) as a percentage."""
    if pd.isna(val):
        return "—"
    return f"{val * 100:.2f}%"


def _fmt_currency(val: float) -> str:
    if pd.isna(val):
        return "—"
    return f"${val:,.0f}"


def _fmt_integer(val: float) -> str:
    if pd.isna(val):
        return "—"
    return f"{val:,.0f}"


def _to_image(fig: go.Figure, width: int = 900, height: int = 500) -> bytes:
    """Export a Plotly figure to PNG bytes via kaleido."""
    return fig.to_image(format="png", width=width, height=height, scale=2, engine="kaleido")


def _month_label(month_str: str | None) -> str:
    """Convert '2026-04-01' to 'April 2026'."""
    if not month_str:
        today = date.today()
        month_str = date(today.year, today.month, 1).isoformat()
    d = date.fromisoformat(month_str)
    return d.strftime("%B %Y")


def _date_range_label(month_str: str | None, as_of_str: str | None) -> str:
    """Convert month + as_of to e.g. 'Apr 1–14'."""
    if not month_str:
        today = date.today()
        month_str = date(today.year, today.month, 1).isoformat()
    if not as_of_str:
        as_of_str = (date.today() - timedelta(days=1)).isoformat()
    m = date.fromisoformat(month_str)
    a = date.fromisoformat(as_of_str)
    return f"{m.strftime('%b')} {m.day}–{a.day}"


def _prior_month(month_str: str | None) -> str:
    """Return the first day of the prior month as ISO string."""
    if not month_str:
        today = date.today()
        month_str = date(today.year, today.month, 1).isoformat()
    m = date.fromisoformat(month_str)
    if m.month == 1:
        return date(m.year - 1, 12, 1).isoformat()
    return date(m.year, m.month - 1, 1).isoformat()


def _prior_date_range_label(month_str: str | None, as_of_str: str | None) -> str:
    """Build the prior-month equivalent date range label."""
    if not month_str:
        today = date.today()
        month_str = date(today.year, today.month, 1).isoformat()
    if not as_of_str:
        as_of_str = (date.today() - timedelta(days=1)).isoformat()
    m = date.fromisoformat(month_str)
    a = date.fromisoformat(as_of_str)
    if m.month == 1:
        prior_m = date(m.year - 1, 12, 1)
    else:
        prior_m = date(m.year, m.month - 1, 1)
    prior_day = min(a.day, 28)
    return f"{prior_m.strftime('%b')} {prior_m.day}–{prior_day}"


# ---------------------------------------------------------------------------
# Chart 1: Pacing Snapshot Table
# ---------------------------------------------------------------------------

_METRIC_ORDER = [
    "sessions", "site_rr", "site_conversion_rate", "phone_vc", "phone_revenue",
    "cart_rr", "cart_conversion_rate", "cart_vc", "cart_revenue",
    "total_vc", "total_revenue",
]

_METRIC_LABELS = {
    "sessions": "Sessions",
    "site_rr": "Site RR",
    "site_conversion_rate": "Site Conversion",
    "phone_vc": "Phone VC",
    "phone_revenue": "Phone Revenue",
    "cart_rr": "Cart RR",
    "cart_conversion_rate": "Cart Conversion",
    "cart_vc": "Cart VC",
    "cart_revenue": "Cart Revenue",
    "total_vc": "VC",
    "total_revenue": "Revenue",
}

_RATE_METRICS = {"site_rr", "site_conversion_rate", "phone_vc", "cart_rr",
                 "cart_conversion_rate", "cart_vc", "total_vc"}
_CURRENCY_METRICS = {"phone_revenue", "cart_revenue", "total_revenue"}


def _fmt_level(metric: str, val: float) -> str:
    """Format a Pacing or Plan level value."""
    if pd.isna(val):
        return "—"
    if metric in _RATE_METRICS:
        return _fmt_rate(val)
    if metric in _CURRENCY_METRICS:
        return _fmt_currency(val)
    return _fmt_integer(val)


def render_pacing_table(
    df: pd.DataFrame,
    month: str | None = None,
    as_of: str | None = None,
) -> bytes:
    """
    Render the full-funnel pacing table as a styled PNG.

    Expects a DataFrame with columns: performance_view, sessions, site_rr,
    site_conversion_rate, phone_vc, phone_revenue, cart_rr, cart_conversion_rate,
    cart_vc, cart_revenue, total_vc, total_revenue (one row per view).
    """
    rows_by_view = {}
    for _, row in df.iterrows():
        rows_by_view[row["performance_view"]] = row

    metric_labels = []
    pacing_vals = []
    plan_vals = []
    vs_plan_vals = []
    mom_vals = []
    yoy_vals = []

    vs_plan_bg = []
    mom_bg = []
    yoy_bg = []
    vs_plan_tc = []
    mom_tc = []
    yoy_tc = []

    for metric in _METRIC_ORDER:
        metric_labels.append(_METRIC_LABELS[metric])

        p_row = rows_by_view.get("Pacing")
        pl_row = rows_by_view.get("Plan")
        vp_row = rows_by_view.get("vs_plan")
        m_row = rows_by_view.get("MoM")
        y_row = rows_by_view.get("YoY")

        pacing_vals.append(_fmt_level(metric, p_row[metric] if p_row is not None else np.nan))
        plan_vals.append(_fmt_level(metric, pl_row[metric] if pl_row is not None else np.nan))

        vp_val = vp_row[metric] if vp_row is not None else np.nan
        m_val = m_row[metric] if m_row is not None else np.nan
        y_val = y_row[metric] if y_row is not None else np.nan

        vs_plan_vals.append(_fmt_pct(vp_val))
        mom_vals.append(_fmt_pct(m_val))
        yoy_vals.append(_fmt_pct(y_val))

        vs_plan_bg.append(_delta_bg(vp_val))
        mom_bg.append(_delta_bg(m_val))
        yoy_bg.append(_delta_bg(y_val))

        vs_plan_tc.append(_delta_text_color(vp_val))
        mom_tc.append(_delta_text_color(m_val))
        yoy_tc.append(_delta_text_color(y_val))

    n = len(metric_labels)
    white_list = [_NEUTRAL_BG] * n
    dark_text = ["rgb(30, 30, 30)"] * n
    metric_bg = [_ALT_ROW_BG if i % 2 == 0 else _NEUTRAL_BG for i in range(n)]

    fig = go.Figure(data=[go.Table(
        columnwidth=[140, 100, 100, 80, 80, 80],
        header=dict(
            values=["<b>Metric</b>", "<b>Pacing</b>", "<b>Plan</b>",
                    "<b>vs Plan</b>", "<b>MoM</b>", "<b>YoY</b>"],
            fill_color=_HEADER_BG,
            font=dict(color=_HEADER_TEXT, size=13, family=_FONT_FAMILY),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[metric_labels, pacing_vals, plan_vals,
                    vs_plan_vals, mom_vals, yoy_vals],
            fill_color=[metric_bg, white_list, white_list,
                        vs_plan_bg, mom_bg, yoy_bg],
            font=dict(
                color=[dark_text, dark_text, dark_text,
                       vs_plan_tc, mom_tc, yoy_tc],
                size=12,
                family=_FONT_FAMILY,
            ),
            align=["left", "right", "right", "center", "center", "center"],
            height=30,
        ),
    )])

    range_lbl = _date_range_label(month, as_of)
    title_text = f"SEO Pacing Snapshot — {_month_label(month)} (through {range_lbl})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, family=_FONT_FAMILY)),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
    )

    return _to_image(fig, width=820, height=60 + 36 + 30 * n)


# ---------------------------------------------------------------------------
# Chart 2: Revenue Waterfall
# ---------------------------------------------------------------------------

def render_waterfall(
    df: pd.DataFrame,
    month: str | None = None,
    as_of: str | None = None,
) -> bytes:
    """
    Render the revenue waterfall decomposition as a styled PNG.

    Expects the same pacing DataFrame (needs Pacing and Plan rows).
    Computes sequential substitution internally.
    """
    rows_by_view = {}
    for _, row in df.iterrows():
        rows_by_view[row["performance_view"]] = row

    P = rows_by_view.get("Plan")
    A = rows_by_view.get("Pacing")
    if P is None or A is None:
        logger.warning("Waterfall chart requires both Plan and Pacing rows")
        return b""

    p_sess = float(P["sessions"])
    p_srr = float(P["site_rr"])
    p_sconv = float(P["site_conversion_rate"])
    p_pgcv = float(P["phone_gcv_order"]) if "phone_gcv_order" in P.index else 0
    p_crr = float(P["cart_rr"])
    p_cconv = float(P["cart_conversion_rate"])
    p_cgcv = float(P["cart_gcv_order"]) if "cart_gcv_order" in P.index else 0

    a_sess = float(A["sessions"])
    a_srr = float(A["site_rr"])
    a_sconv = float(A["site_conversion_rate"])
    a_pgcv = float(A["phone_gcv_order"]) if "phone_gcv_order" in A.index else 0
    a_crr = float(A["cart_rr"])
    a_cconv = float(A["cart_conversion_rate"])
    a_cgcv = float(A["cart_gcv_order"]) if "cart_gcv_order" in A.index else 0

    plan_phone = p_sess * p_srr * p_sconv * p_pgcv
    plan_cart = p_sess * p_crr * p_cconv * p_cgcv
    plan_rev = plan_phone + plan_cart

    after_sess_phone = a_sess * p_srr * p_sconv * p_pgcv
    after_sess_cart = a_sess * p_crr * p_cconv * p_cgcv
    sessions_impact = (after_sess_phone + after_sess_cart) - plan_rev

    after_srr_phone = a_sess * a_srr * p_sconv * p_pgcv
    srr_impact = after_srr_phone - after_sess_phone

    after_sconv_phone = a_sess * a_srr * a_sconv * p_pgcv
    sconv_impact = after_sconv_phone - after_srr_phone

    actual_phone = a_sess * a_srr * a_sconv * a_pgcv
    pgcv_impact = actual_phone - after_sconv_phone

    after_crr_cart = a_sess * a_crr * p_cconv * p_cgcv
    crr_impact = after_crr_cart - after_sess_cart

    after_cconv_cart = a_sess * a_crr * a_cconv * p_cgcv
    cconv_impact = after_cconv_cart - after_crr_cart

    actual_cart = a_sess * a_crr * a_cconv * a_cgcv
    cgcv_impact = actual_cart - after_cconv_cart

    pacing_rev = actual_phone + actual_cart

    labels = [
        "Plan Revenue", "Sessions", "Site RR", "Site Conv.",
        "Phone GCV/Order", "Cart RR", "Cart Conv.", "Cart GCV/Order",
        "Pacing Revenue",
    ]
    values = [
        plan_rev, sessions_impact, srr_impact, sconv_impact,
        pgcv_impact, crr_impact, cconv_impact, cgcv_impact,
        pacing_rev,
    ]
    measures = [
        "absolute", "relative", "relative", "relative",
        "relative", "relative", "relative", "relative",
        "total",
    ]

    bar_colors = []
    for m, v in zip(measures, values):
        if m != "relative":
            bar_colors.append("rgb(44, 62, 80)")
        elif v >= 0:
            bar_colors.append("rgb(39, 174, 96)")
        else:
            bar_colors.append("rgb(231, 76, 60)")

    text_labels = []
    for m, v in zip(measures, values):
        if m == "absolute" or m == "total":
            text_labels.append(f"${v:,.0f}")
        else:
            sign = "+" if v >= 0 else ""
            text_labels.append(f"{sign}${v:,.0f}")

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=measures,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11, family=_FONT_FAMILY),
        connector=dict(line=dict(color="rgb(180,180,180)", width=1)),
        increasing=dict(marker=dict(color="rgb(39, 174, 96)")),
        decreasing=dict(marker=dict(color="rgb(231, 76, 60)")),
        totals=dict(marker=dict(color="rgb(44, 62, 80)")),
    ))

    range_lbl = _date_range_label(month, as_of)
    waterfall_title = f"Revenue Waterfall: Plan → Pacing — {_month_label(month)} ({range_lbl})"

    fig.update_layout(
        title=dict(text=waterfall_title, font=dict(size=16, family=_FONT_FAMILY)),
        yaxis=dict(title="Revenue ($)", tickformat="$,.0f"),
        xaxis=dict(tickangle=-30),
        margin=dict(l=80, r=30, t=60, b=100),
        paper_bgcolor="white",
        plot_bgcolor="rgb(250, 250, 250)",
        showlegend=False,
    )

    return _to_image(fig, width=900, height=500)


# ---------------------------------------------------------------------------
# Chart 3: Top-of-Funnel Performance
# ---------------------------------------------------------------------------

def render_tof_chart(
    df: pd.DataFrame,
    month: str | None = None,
    as_of: str | None = None,
) -> bytes:
    """
    Render a GSC top-of-funnel comparison table as a styled PNG.

    Expects a DataFrame with columns: period, clicks, impressions, ctr_pct, avg_rank
    and rows for 'Current' and 'Prior'.
    """
    rows_by_period = {}
    for _, row in df.iterrows():
        rows_by_period[row["period"]] = row

    cur = rows_by_period.get("Current")
    pri = rows_by_period.get("Prior")
    if cur is None or pri is None:
        logger.warning("TOF chart requires both Current and Prior rows")
        return b""

    metrics = [
        ("Clicks", "clicks", "int"),
        ("Impressions", "impressions", "int"),
        ("CTR", "ctr_pct", "pct"),
        ("Avg Rank", "avg_rank", "rank"),
    ]

    metric_labels = []
    current_vals = []
    prior_vals = []
    delta_vals = []
    delta_pct_vals = []
    delta_bg_colors = []
    delta_text_colors = []

    for label, col, fmt_type in metrics:
        c_val = float(cur[col])
        p_val = float(pri[col])
        delta = c_val - p_val
        delta_frac = (c_val / p_val - 1) if p_val != 0 else 0

        metric_labels.append(label)

        if fmt_type == "int":
            current_vals.append(f"{c_val:,.0f}")
            prior_vals.append(f"{p_val:,.0f}")
            sign = "+" if delta >= 0 else ""
            delta_vals.append(f"{sign}{delta:,.0f}")
        elif fmt_type == "pct":
            current_vals.append(f"{c_val:.2f}%")
            prior_vals.append(f"{p_val:.2f}%")
            sign = "+" if delta >= 0 else ""
            delta_vals.append(f"{sign}{delta:.2f}pp")
        elif fmt_type == "rank":
            current_vals.append(f"{c_val:.1f}")
            prior_vals.append(f"{p_val:.1f}")
            rank_delta = p_val - c_val  # lower rank = better, so flip sign for color
            sign = "+" if delta >= 0 else ""
            delta_vals.append(f"{sign}{delta:.1f}")
            delta_frac = rank_delta / p_val if p_val != 0 else 0

        delta_pct_vals.append(_fmt_pct(delta_frac))
        delta_bg_colors.append(_delta_bg(delta_frac))
        delta_text_colors.append(_delta_text_color(delta_frac))

    n = len(metric_labels)
    white_list = [_NEUTRAL_BG] * n
    dark_text = ["rgb(30, 30, 30)"] * n
    metric_bg = [_ALT_ROW_BG if i % 2 == 0 else _NEUTRAL_BG for i in range(n)]

    cur_label = _date_range_label(month, as_of)
    prior_label = _prior_date_range_label(month, as_of)

    fig = go.Figure(data=[go.Table(
        columnwidth=[120, 100, 100, 90, 80],
        header=dict(
            values=["<b>Metric</b>", f"<b>{cur_label}</b>", f"<b>{prior_label}</b>",
                    "<b>Delta</b>", "<b>Delta %</b>"],
            fill_color=_HEADER_BG,
            font=dict(color=_HEADER_TEXT, size=13, family=_FONT_FAMILY),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[metric_labels, current_vals, prior_vals,
                    delta_vals, delta_pct_vals],
            fill_color=[metric_bg, white_list, white_list,
                        delta_bg_colors, delta_bg_colors],
            font=dict(
                color=[dark_text, dark_text, dark_text,
                       delta_text_colors, delta_text_colors],
                size=12,
                family=_FONT_FAMILY,
            ),
            align=["left", "right", "right", "right", "center"],
            height=30,
        ),
    )])

    tof_title = f"Top-of-Funnel: Search Performance — {_month_label(month)} vs {_month_label(_prior_month(month))}"

    fig.update_layout(
        title=dict(text=tof_title, font=dict(size=16, family=_FONT_FAMILY)),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
    )

    return _to_image(fig, width=680, height=60 + 36 + 30 * n)
