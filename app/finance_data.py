"""
Finance data layer.

Uses the finance_query for daily actuals and rpt_texas_daily_pacing
for plan/pacing data. Finance queries are the source of truth for
performance reporting; session-level queries remain directional only.

Field mapping assumptions (documented inline):
- Plan table channel 'PMAX' maps to finance/UI channel 'pMax'
- Finance query merges Social+Display→'Social' and Other+Internal+Referral→'Other';
  plan table keeps them separate, so we expand when filtering plan data.
- ZLUR and G2C are not present in the plan table; populated from finance MTD instead.
- VC excludes SERP orders (matches plan_query logic: (total_orders - serp_orders)/sessions).
"""

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.db import get_connection as _get_connection

# Finance-query channel name → plan table equivalents
_CHANNEL_EXPAND = {
    "pMax": ["PMAX"],
    "Organic": ["Organic", "SEO"],
    "Social": ["Social", "Display"],
    "Other": ["Other", "Internal", "Referral"],
}


# ── data fetching ──────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner="Loading finance actuals…")
def fetch_finance_daily() -> pd.DataFrame:
    """Execute the finance_query file and return daily data by channel."""
    query_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "queries", "finance_query"
    )
    with open(query_path) as f:
        raw = f.read()
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(raw)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)

    for col in ["TheDate", "WeekBeginning"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

    # Databricks returns decimal.Decimal — force everything numeric to float
    for c in df.columns:
        if c not in ("TheDate", "WeekBeginning", "MarketingChannel", "calendar_year_month"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    return df


@st.cache_data(ttl=1800, show_spinner="Loading plan/pacing data…")
def fetch_plan_pacing() -> pd.DataFrame:
    """Query rpt_texas_daily_pacing for Pacing and Plan views.

    Two historical gotchas to know about:

    1. **Channel rename.** Pre-2026-01-01 the table uses ``MarketingChannel='Organic'``;
       from 2026-01-01 onward it uses ``'SEO'``. The ``_CHANNEL_EXPAND`` map
       already translates our UI label ``"Organic"`` to both values.

    2. **Plan view rename.** Pre-2026-01-01 the plan row is ``performance_view='Plan'``;
       from 2026-01-01 onward Finance publishes two views — ``'Final'`` (locked)
       and ``'Internal Plan'`` (live / refreshed). We accept BOTH and prefer
       ``'Final'`` when present, falling back to ``'Internal Plan'`` — matching
       the Finance team's canonical pacing query.

    3. **In-flight month pacing.** The table stores *cumulative MTD* values on
       each daily row of ``performance_view='Pacing'``, so a single
       ``rpt_date = <latest>`` row tells you month-to-date totals. We used to
       hardcode ``rpt_date = CURRENT_DATE - 1`` but the table loads ~1 day
       behind the warehouse clock, so that returned zero rows and silently
       blanked the Pacing / Plan / vs-Plan columns. We now bind to the actual
       ``MAX(rpt_date)`` of the Pacing view.
    """
    query = """
    WITH pacing_horizon AS (
        SELECT MAX(rpt_date) AS max_rpt_date
        FROM energy_prod.energy.rpt_texas_daily_pacing
        WHERE performance_view = 'Pacing'
    ),
    raw AS (
        SELECT
            tx.rpt_date,
            tx.performance_view,
            tx.MarketingChannel,
            COALESCE(tx.sessions, 0)           AS sessions,
            COALESCE(tx.cart_entries, 0)       AS cart_entries,
            COALESCE(tx.cart_orders, 0)        AS cart_orders,
            COALESCE(tx.phone_orders, 0)       AS phone_orders,
            COALESCE(tx.total_orders, 0)       AS total_orders,
            COALESCE(tx.serp_orders, 0)        AS serp_orders,
            COALESCE(tx.site_queue_calls, 0)   AS site_queue_calls,
            COALESCE(tx.site_phone_orders, 0)  AS site_phone_orders,
            COALESCE(tx.phone_revenue, 0)      AS phone_revenue,
            COALESCE(tx.cart_revenue, 0)       AS cart_revenue,
            COALESCE(tx.revenue, 0)            AS revenue
        FROM energy_prod.energy.rpt_texas_daily_pacing tx
        CROSS JOIN pacing_horizon ph
        WHERE tx.rpt_date >= '2026-01-01'
          AND (
              tx.performance_view IN ('Pacing', 'Final', 'Internal Plan')
          )
          AND (
              -- Past months: take the locked month-end snapshot
              (tx.rpt_date = last_day(tx.rpt_date)
               AND date_trunc('month', tx.rpt_date) != date_trunc('month', ph.max_rpt_date))
              -- In-flight month: take the latest fully-loaded daily row
              OR tx.rpt_date = ph.max_rpt_date
          )
    ),
    -- Collapse 'Final' and 'Internal Plan' into a single 'Plan' view,
    -- preferring 'Final' where it exists (any rpt_date × MarketingChannel
    -- that has a Final row drops the Internal Plan duplicate).
    plan_dedup AS (
        SELECT
            rpt_date,
            'Plan' AS perf_view,
            MarketingChannel,
            sessions, cart_entries, cart_orders, phone_orders, total_orders,
            serp_orders, site_queue_calls, site_phone_orders,
            phone_revenue, cart_revenue, revenue,
            ROW_NUMBER() OVER (
                PARTITION BY rpt_date, MarketingChannel
                ORDER BY CASE performance_view
                             WHEN 'Final' THEN 0
                             WHEN 'Internal Plan' THEN 1
                             ELSE 2
                         END
            ) AS rn
        FROM raw
        WHERE performance_view IN ('Final', 'Internal Plan')
    )
    SELECT rpt_date, 'Pacing' AS perf_view, MarketingChannel,
           sessions, cart_entries, cart_orders, phone_orders, total_orders,
           serp_orders, site_queue_calls, site_phone_orders,
           phone_revenue, cart_revenue, revenue
      FROM raw
     WHERE performance_view = 'Pacing'
    UNION ALL
    SELECT rpt_date, perf_view, MarketingChannel,
           sessions, cart_entries, cart_orders, phone_orders, total_orders,
           serp_orders, site_queue_calls, site_phone_orders,
           phone_revenue, cart_revenue, revenue
      FROM plan_dedup
     WHERE rn = 1
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    if "rpt_date" in df.columns:
        df["rpt_date"] = pd.to_datetime(df["rpt_date"]).dt.date
    for c in df.columns:
        if c not in ("rpt_date", "perf_view", "MarketingChannel"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    return df


# ── channel mapping ────────────────────────────────────────────────────────

def _map_channels_for_plan(channels: list[str]) -> list[str]:
    """Expand finance/UI channel names to plan table equivalents."""
    result: list[str] = []
    for ch in channels:
        expanded = _CHANNEL_EXPAND.get(ch)
        if expanded:
            result.extend(expanded)
        else:
            result.append(ch)
    return result


# ── aggregation helpers ────────────────────────────────────────────────────

_FIN_SUM_COLS = [
    "Total_Sessions", "ZipEntries", "CartEntries", "TotalCartOrders",
    "TotalPhoneOrders", "site_phone_orders", "site_queue_calls",
    "Serp_Orders", "TotalCartGCV", "TotalPhoneGCV",
]

_PLAN_SUM_COLS = [
    "sessions", "cart_entries", "cart_orders", "phone_orders",
    "total_orders", "serp_orders", "site_queue_calls",
    "site_phone_orders", "cart_revenue", "phone_revenue",
]


def _agg(df: pd.DataFrame, cols: list[str]) -> dict:
    return {c: df[c].sum() if c in df.columns else 0 for c in cols}


def _rates_finance(a: dict) -> dict:
    """Compute funnel metrics from aggregated finance daily data.

    Phone revenue from the finance query (`TotalPhoneGCV`) already includes
    Site + SERP + cross-sell, which matches the plan table's `phone_revenue`.
    Total phone orders (`TotalPhoneOrders`) also include both Site and SERP.
    """
    s   = a["Total_Sessions"]
    z   = a["ZipEntries"]
    ce  = a["CartEntries"]
    co  = a["TotalCartOrders"]
    spo = a["site_phone_orders"]
    sqc = a["site_queue_calls"]
    tpo = a["TotalPhoneOrders"]          # Site + SERP phone orders
    pg  = a["TotalPhoneGCV"]             # Site + SERP + cross-sell phone revenue
    cg  = a["TotalCartGCV"]
    return {
        "LP Sessions":       s,
        "Phone RR":          sqc / s  if s  else None,
        "Phone VC":          spo / s  if s  else None,
        "ZLUR":              z / s    if s  else None,
        "G2C":               ce / z   if z  else None,
        "Cart RR":           ce / s   if s  else None,
        "Cart Conversion":   co / ce  if ce else None,
        "Cart VC":           co / s   if s  else None,
        "VC":                (co + spo) / s if s else None,
        "Phone Rev/Order":   pg / tpo if tpo else None,
        "Cart Rev/Order":    cg / co  if co  else None,
        "Phone Revenue":     pg,
        "Cart Revenue":      cg,
        "Revenue":           pg + cg,
    }


def _rates_plan(a: dict) -> dict:
    """Compute funnel metrics from aggregated plan/pacing data.

    Plan table phone_orders/phone_revenue are Site + SERP (matches finance).
    """
    s   = a["sessions"]
    ce  = a["cart_entries"]
    co  = a["cart_orders"]
    spo = a["site_phone_orders"]
    sqc = a["site_queue_calls"]
    tpo = a["phone_orders"]              # plan table: Site + SERP combined
    pr  = a["phone_revenue"]
    cr  = a["cart_revenue"]
    return {
        "LP Sessions":       s,
        "Phone RR":          sqc / s  if s  else None,
        "Phone VC":          spo / s  if s  else None,
        "ZLUR":              None,
        "G2C":               None,
        "Cart RR":           ce / s   if s  else None,
        "Cart Conversion":   co / ce  if ce else None,
        "Cart VC":           co / s   if s  else None,
        "VC":                (co + spo) / s if s else None,
        "Phone Rev/Order":   pr / tpo if tpo else None,
        "Cart Rev/Order":    cr / co  if co  else None,
        "Phone Revenue":     pr,
        "Cart Revenue":      cr,
        "Revenue":           pr + cr,
    }


def _pct_delta(curr, prior):
    if curr is None or prior is None or prior == 0:
        return None
    return (curr - prior) / abs(prior)


# ── row definitions (matches the example image layout) ─────────────────────
# Each entry: (section_header_or_None, metric_name_or_None, type, footnote_key_or_None)
# `footnote_key` is matched against `_FOOTNOTES` in render_summary_html to
# attach a superscript asterisk + a table footer note.

_ROWS: list[tuple] = [
    (None,                       "LP Sessions",       "volume",  None),
    ("Phone Funnel (Site Only)", None,                None,      None),
    (None,                       "Phone RR",          "rate",    None),
    (None,                       "Phone VC",          "rate",    None),
    ("Cart Funnel",              None,                None,      None),
    (None,                       "ZLUR",              "rate",    None),
    (None,                       "G2C",               "rate",    None),
    (None,                       "Cart RR",           "rate",    None),
    (None,                       "Cart Conversion",   "rate",    None),
    (None,                       "Cart VC",           "rate",    None),
    ("Bottom Line",              None,                None,      None),
    (None,                       "VC",                "rate",    None),
    (None,                       "Phone Rev/Order",   "dollar",  "site_plus_serp"),
    (None,                       "Cart Rev/Order",    "dollar",  None),
    (None,                       "Phone Revenue",     "dollar",  "site_plus_serp"),
    (None,                       "Cart Revenue",      "dollar",  None),
    (None,                       "Revenue",           "dollar",  None),
]

# Footnote text keyed by the 4th column of `_ROWS`.
_FOOTNOTES: dict[str, str] = {
    "site_plus_serp": "Phone metrics include both Site and SERP orders/revenue.",
}


# ── main builder ───────────────────────────────────────────────────────────

def build_funnel_summary(
    finance_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    channels: list[str],
    curr_start: date | None = None,
    curr_end: date | None = None,
    prior_start: date | None = None,
    prior_end: date | None = None,
) -> list[dict]:
    """
    Build the funnel summary table rows.

    The current/prior period dates come from the sidebar controls so the
    "current week" column and WoW/P4WA are aligned with whatever period
    the user selected.

    Returns a list of dicts. Section-header rows have a ``section`` key;
    metric rows have ``metric``, ``type``, and value/delta keys for each
    column of the summary table.
    """
    today = date.today()

    # Filter finance data to selected channels
    fin = (
        finance_df[finance_df["MarketingChannel"].isin(channels)].copy()
        if channels else finance_df.copy()
    )

    # ── MTD (used for ZLUR/G2C pacing since plan table lacks them) ──
    m1 = today.replace(day=1)
    mtd = fin[(fin["TheDate"] >= m1) & (fin["TheDate"] < today)]
    mtd_r = _rates_finance(_agg(mtd, _FIN_SUM_COLS))

    # ── Current / Prior periods from sidebar ──
    if curr_start and curr_end:
        curr_df = fin[(fin["TheDate"] >= curr_start) & (fin["TheDate"] <= curr_end)]
    else:
        curr_monday = today - timedelta(days=today.weekday())
        cw = curr_monday - timedelta(weeks=1)
        curr_df = fin[fin["WeekBeginning"] == cw]

    if prior_start and prior_end:
        prior_df = fin[(fin["TheDate"] >= prior_start) & (fin["TheDate"] <= prior_end)]
    else:
        curr_monday = today - timedelta(days=today.weekday())
        pw = curr_monday - timedelta(weeks=2)
        prior_df = fin[fin["WeekBeginning"] == pw]

    curr_r  = _rates_finance(_agg(curr_df, _FIN_SUM_COLS)) if not curr_df.empty else {}
    prior_r = _rates_finance(_agg(prior_df, _FIN_SUM_COLS)) if not prior_df.empty else {}

    # ── P4WA: same day-of-week pattern from the 4 weeks before the prior week ──
    if curr_start and curr_end:
        # Determine which weekdays are in the current period (0=Mon … 6=Sun)
        _valid_wdays = set()
        d = curr_start
        while d <= curr_end:
            _valid_wdays.add(d.weekday())
            d += timedelta(days=1)

        # Go back 4 weekly windows before the prior period, keeping only
        # the same weekdays so WTD comparisons are apples-to-apples.
        _prior_monday = (prior_start or curr_start) - timedelta(
            days=(prior_start or curr_start).weekday()
        )
        p4_dates = []
        for wk in range(1, 5):
            wk_monday = _prior_monday - timedelta(weeks=wk)
            for wd in _valid_wdays:
                p4_dates.append(wk_monday + timedelta(days=wd))
        p4_pool = fin[fin["TheDate"].isin(p4_dates)]
    else:
        curr_monday = today - timedelta(days=today.weekday())
        p4_starts_list = [curr_monday - timedelta(weeks=w) for w in range(2, 6)]
        p4_pool = fin[fin["WeekBeginning"].isin(p4_starts_list)]

    if not p4_pool.empty:
        p4_r = _rates_finance(_agg(p4_pool, _FIN_SUM_COLS))
        # Volume / dollar totals need to be averaged across the 4 prior weeks
        # for a true "per-week" comparison. Rate metrics are already normalized
        # to per-session ratios and stay as-is.
        for _vol_metric in ("LP Sessions", "Cart Revenue", "Phone Revenue", "Revenue"):
            if p4_r.get(_vol_metric) is not None:
                p4_r[_vol_metric] = p4_r[_vol_metric] / 4
    else:
        p4_r = {}

    # ── Plan / Pacing from rpt_texas_daily_pacing ──
    pch = _map_channels_for_plan(channels)
    pm = plan_df[(plan_df["rpt_date"] >= m1) & (plan_df["rpt_date"] <= today)]

    pac = pm[(pm["perf_view"] == "Pacing") & (pm["MarketingChannel"].isin(pch))]
    pln = pm[(pm["perf_view"] == "Plan")   & (pm["MarketingChannel"].isin(pch))]

    pac_r = _rates_plan(_agg(pac, _PLAN_SUM_COLS)) if not pac.empty else {}
    pln_r = _rates_plan(_agg(pln, _PLAN_SUM_COLS)) if not pln.empty else {}

    if pac_r:
        pac_r["ZLUR"] = mtd_r.get("ZLUR")
        pac_r["G2C"]  = mtd_r.get("G2C")

    # ── Assemble rows ──
    rows: list[dict] = []
    for section, metric, mtype, note_key in _ROWS:
        if section:
            rows.append({"section": section})
            continue
        rows.append({
            "metric": metric,
            "type": mtype,
            "note_key": note_key,
            "march_pacing":  pac_r.get(metric) if pac_r else None,
            "plan_actual":   pln_r.get(metric) if pln_r else None,
            "v_plan":        _pct_delta(pac_r.get(metric), pln_r.get(metric)) if pac_r and pln_r else None,
            "current_week":  curr_r.get(metric) if curr_r else None,
            "wow":           _pct_delta(curr_r.get(metric), prior_r.get(metric)) if curr_r and prior_r else None,
            "p4wa":          _pct_delta(curr_r.get(metric), p4_r.get(metric)) if curr_r and p4_r else None,
        })
    return rows


# ── formatting / rendering ─────────────────────────────────────────────────

def _fmt_val(val, mtype):
    if val is None:
        return ""
    if mtype == "volume":
        return f"{int(round(val)):,}"
    if mtype == "rate":
        pct = val * 100
        s = f"{pct:.2f}".rstrip("0").rstrip(".")
        return s + "%"
    if mtype == "dollar":
        # Per-order values (< $1k) show cents for precision; totals show as
        # whole dollars with thousands separators.
        if abs(val) >= 1000:
            return f"${val:,.0f}"
        return f"${val:,.2f}"
    return str(val)


def _fmt_delta(val):
    if val is None:
        return ""
    return f"{round(val * 100)}%"


def _delta_style(val):
    """Inline CSS for conditional coloring of delta cells.
    ±2.5% or less is treated as flat (yellow); beyond that is green/red."""
    if val is None:
        return ""
    if val > 0.025:
        return "background-color:#c6efce;color:#006100;"
    if val < -0.025:
        return "background-color:#ffc7ce;color:#9c0006;"
    if val != 0:
        return "background-color:#ffeb9c;color:#9c6500;"
    return ""


_FOOTNOTE_MARKERS = ["*", "†", "‡", "§"]


def render_summary_html(rows: list[dict], report_date: date, period_label: str | None = None) -> str:
    """Render the funnel summary as an HTML table styled to match the example.

    Rows whose ``note_key`` matches an entry in ``_FOOTNOTES`` get a
    superscript marker (``*``, ``†``, …) after the metric name, and the
    corresponding note is rendered beneath the table.
    """
    month = report_date.strftime("%B")
    ds = period_label or report_date.strftime("%-m/%-d/%y")

    # Assign a stable marker to each distinct note_key that actually appears.
    keys_in_use: list[str] = []
    for row in rows:
        k = row.get("note_key")
        if k and k in _FOOTNOTES and k not in keys_in_use:
            keys_in_use.append(k)
    note_markers = {k: _FOOTNOTE_MARKERS[i % len(_FOOTNOTE_MARKERS)]
                    for i, k in enumerate(keys_in_use)}

    hdr_style = "padding:10px 12px;border:1px solid #bbb;"
    html = (
        '<table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:14px;">'
        "<thead>"
        '<tr style="background:#2c3e50;color:white;text-align:center;">'
        f'<th style="{hdr_style}text-align:left;min-width:180px;font-size:15px;">Core Funnel</th>'
        f'<th style="{hdr_style}">{month} Pacing</th>'
        f'<th style="{hdr_style}">Plan Actual</th>'
        f'<th style="{hdr_style}">V. Plan</th>'
        f'<th style="{hdr_style}">{ds}</th>'
        f'<th style="{hdr_style}">WoW</th>'
        f'<th style="{hdr_style}">P4WA</th>'
        "</tr></thead><tbody>"
    )

    cell = "padding:8px 12px;border:1px solid #ddd;text-align:center;"

    for row in rows:
        if "section" in row:
            html += (
                "<tr><td colspan=\"7\" style=\"padding:6px 14px;border:1px solid #ddd;"
                f"font-weight:bold;font-style:italic;background:#eaeaea;\">{row['section']}</td></tr>"
            )
            continue

        m = row["metric"]
        t = row["type"]
        mp  = _fmt_val(row.get("march_pacing"), t)
        pa  = _fmt_val(row.get("plan_actual"), t)
        vp  = _fmt_delta(row.get("v_plan"))
        cw  = _fmt_val(row.get("current_week"), t)
        wow = _fmt_delta(row.get("wow"))
        p4  = _fmt_delta(row.get("p4wa"))

        vs = _delta_style(row.get("v_plan"))
        ws = _delta_style(row.get("wow"))
        ps = _delta_style(row.get("p4wa"))

        marker_html = ""
        nk = row.get("note_key")
        if nk and nk in note_markers:
            marker_html = (
                '<sup style="color:#6b6b6b;font-weight:600;margin-left:2px;">'
                f'{note_markers[nk]}</sup>'
            )

        html += (
            "<tr>"
            f'<td style="padding:8px 14px;border:1px solid #ddd;font-weight:500;">{m}{marker_html}</td>'
            f'<td style="{cell}">{mp}</td>'
            f'<td style="{cell}">{pa}</td>'
            f'<td style="{cell}{vs}">{vp}</td>'
            f'<td style="{cell}">{cw}</td>'
            f'<td style="{cell}{ws}">{wow}</td>'
            f'<td style="{cell}{ps}">{p4}</td>'
            "</tr>"
        )

    html += "</tbody></table>"

    if note_markers:
        notes = "".join(
            f'<div><sup style="color:#6b6b6b;font-weight:600;">{note_markers[k]}</sup> '
            f'{_FOOTNOTES[k]}</div>'
            for k in keys_in_use
        )
        html += (
            '<div style="margin-top:6px;font-family:Arial,sans-serif;font-size:12px;'
            f'color:#6b6b6b;line-height:1.5;">{notes}</div>'
        )

    return html
