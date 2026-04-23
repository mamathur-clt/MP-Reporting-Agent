"""
SEO-specific tools for the Slack reporting agent.

Each tool encapsulates a parameterized Databricks query and returns
structured results the LLM can interpret and narrate.
"""

import json
import logging
from datetime import date, timedelta

from bot.db import execute_readonly_sql

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared SQL fragments
# ---------------------------------------------------------------------------

_URL_TO_PAGE_TYPE_CTE = """
url_to_page_type AS (
  SELECT landing_page, landing_page_type
  FROM (
    SELECT
      RTRIM('/', LOWER(first_page_url)) AS landing_page,
      landing_page_type,
      ROW_NUMBER() OVER (
        PARTITION BY RTRIM('/', LOWER(first_page_url))
        ORDER BY SUM(sessions) DESC
      ) AS rn
    FROM energy_prod.data_science.mp_session_level_query
    WHERE _date >= '2025-01-01'
      AND landing_page_type IS NOT NULL
    GROUP BY RTRIM('/', LOWER(first_page_url)), landing_page_type
  )
  WHERE rn = 1
)"""

_GSC_DOMAINS = (
    "'choosetexaspower.org','saveonenergy.com',"
    "'chooseenergy.com','texaselectricrates.com'"
)

_SITE_CASE = """CASE g.domain
      WHEN 'choosetexaspower.org'  THEN 'CTXP'
      WHEN 'saveonenergy.com'      THEN 'SOE'
      WHEN 'chooseenergy.com'      THEN 'Choose TX'
      WHEN 'texaselectricrates.com' THEN 'TXER'
    END"""

_NORMALIZE_PAGE = """RTRIM('/',
      LOWER(
        CASE WHEN POSITION('#' IN g.page) > 0
          THEN LEFT(g.page, POSITION('#' IN g.page) - 1)
          ELSE g.page
        END
      )
    )"""


def _default_month_start() -> str:
    today = date.today()
    return date(today.year, today.month, 1).isoformat()


def _default_as_of_date() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def _prior_month_start(month_start: str) -> str:
    d = date.fromisoformat(month_start)
    if d.month == 1:
        return date(d.year - 1, 12, 1).isoformat()
    return date(d.year, d.month - 1, 1).isoformat()


def _same_day_prior_month(as_of: str, month_start: str) -> str:
    """Same day-of-month in the prior month for apples-to-apples comparison."""
    d = date.fromisoformat(as_of)
    prior_start = date.fromisoformat(_prior_month_start(month_start))
    day = min(d.day, 28)
    return date(prior_start.year, prior_start.month, day).isoformat()


_gsc_freshness_cache: dict[str, str] = {}


def _resolve_gsc_as_of(month: str, requested_as_of: str) -> str:
    """Return the latest GSC date with actual data, capped at requested_as_of.

    GSC data typically lags 2-3 days. Using yesterday blindly causes
    artificial declines because recent days have no data yet.
    """
    cache_key = f"{month}:{requested_as_of}"
    if cache_key in _gsc_freshness_cache:
        return _gsc_freshness_cache[cache_key]

    sql = f"""
SELECT MAX(date) AS latest_date
FROM lakehouse_production.common.gsc_search_analytics_d_1
WHERE domain IN ({_GSC_DOMAINS})
  AND impressions > 0
  AND date >= '{month}'
  AND date <= '{requested_as_of}'
"""
    result_str, df = execute_readonly_sql(sql, max_rows=1)
    resolved = requested_as_of
    if df is not None and not df.empty and df.iloc[0]["latest_date"] is not None:
        latest = df.iloc[0]["latest_date"]
        if hasattr(latest, "isoformat"):
            resolved = latest.date().isoformat() if hasattr(latest, "date") else latest.isoformat()
        else:
            resolved = str(latest)[:10]
    if resolved != requested_as_of:
        logger.info("GSC freshness: adjusted as_of from %s to %s", requested_as_of, resolved)
    _gsc_freshness_cache[cache_key] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Tool 1: SEO Finance Pacing
# ---------------------------------------------------------------------------

def _build_pacing_sql(month: str, as_of: str) -> str:
    """Build the full SEO finance pacing query with date parameters inlined."""
    return f"""
WITH q AS (
  WITH tx_filtered AS (
    SELECT
      CAST(tx.rpt_date AS DATE) AS date, tx.month,
      CASE WHEN tx.performance_view = 'Final' THEN 'Plan' ELSE tx.performance_view END AS performance_view,
      CASE WHEN CAST(tx.rpt_date AS DATE) < '2026-01-01' AND tx.MarketingChannel = 'Organic' THEN 'SEO' ELSE tx.MarketingChannel END AS MarketingChannel,
      tx.sessions, tx.site_queue_calls, tx.site_gross_calls, tx.site_phone_orders,
      tx.phone_orders, tx.cart_entries, tx.cart_orders, tx.total_orders,
      tx.phone_revenue, tx.cart_revenue, tx.revenue, tx.serp_orders
    FROM energy_prod.energy.rpt_texas_daily_pacing tx
    WHERE tx.performance_view IN ('Pacing', 'Final', 'Internal Plan')
      AND CAST(tx.rpt_date AS DATE) >= '2025-01-01'
      AND array_contains(ARRAY('SEO'),
        CASE WHEN CAST(tx.rpt_date AS DATE) < '2026-01-01' AND tx.MarketingChannel = 'Organic' THEN 'SEO' ELSE tx.MarketingChannel END
      )
  ),
  agg AS (
    SELECT date, month, performance_view,
      SUM(sessions) AS sessions,
      SUM(site_queue_calls) AS site_queue_calls,
      SUM(site_gross_calls) AS site_gross_calls,
      SUM(site_phone_orders) AS site_phone_orders,
      SUM(phone_orders) AS phone_orders,
      SUM(cart_entries) AS cart_entries,
      SUM(cart_orders) AS cart_orders,
      SUM(total_orders) AS total_orders,
      SUM(phone_revenue) AS phone_revenue,
      SUM(cart_revenue) AS cart_revenue,
      SUM(revenue) AS total_revenue,
      SUM(serp_orders) AS serp_orders
    FROM tx_filtered GROUP BY 1,2,3
  ),
  cte_base AS (
    SELECT date, month, performance_view,
      sessions,
      site_queue_calls * 1.0 / NULLIF(sessions,0) AS site_rr,
      site_phone_orders * 1.0 / NULLIF(site_queue_calls,0) AS site_conversion_rate,
      site_phone_orders * 1.0 / NULLIF(sessions,0) AS phone_vc,
      phone_revenue,
      cart_entries * 1.0 / NULLIF(sessions,0) AS cart_rr,
      cart_orders * 1.0 / NULLIF(cart_entries,0) AS cart_conversion_rate,
      cart_orders * 1.0 / NULLIF(sessions,0) AS cart_vc,
      cart_revenue,
      (total_orders - serp_orders) * 1.0 / NULLIF(sessions,0) AS total_vc,
      total_revenue,
      phone_revenue * 1.0 / NULLIF(phone_orders,0) AS phone_gcv_order,
      cart_revenue * 1.0 / NULLIF(cart_orders,0) AS cart_gcv_order
    FROM agg
  ),
  cte_vs_plan AS (
    SELECT p.date, p.month, 'vs_plan' AS performance_view,
      (p.sessions/NULLIF(pl.sessions,0))-1 AS sessions,
      (p.site_rr/NULLIF(pl.site_rr,0))-1 AS site_rr,
      (p.site_conversion_rate/NULLIF(pl.site_conversion_rate,0))-1 AS site_conversion_rate,
      (p.phone_vc/NULLIF(pl.phone_vc,0))-1 AS phone_vc,
      (p.phone_revenue/NULLIF(pl.phone_revenue,0))-1 AS phone_revenue,
      (p.cart_rr/NULLIF(pl.cart_rr,0))-1 AS cart_rr,
      (p.cart_conversion_rate/NULLIF(pl.cart_conversion_rate,0))-1 AS cart_conversion_rate,
      (p.cart_vc/NULLIF(pl.cart_vc,0))-1 AS cart_vc,
      (p.cart_revenue/NULLIF(pl.cart_revenue,0))-1 AS cart_revenue,
      (p.total_vc/NULLIF(pl.total_vc,0))-1 AS total_vc,
      (p.total_revenue/NULLIF(pl.total_revenue,0))-1 AS total_revenue,
      (p.phone_gcv_order/NULLIF(pl.phone_gcv_order,0))-1 AS phone_gcv_order,
      (p.cart_gcv_order/NULLIF(pl.cart_gcv_order,0))-1 AS cart_gcv_order
    FROM cte_base p
    LEFT JOIN cte_base pl ON pl.date=p.date AND pl.month=p.month AND pl.performance_view='Plan'
    WHERE p.performance_view='Pacing'
  ),
  cte_mom AS (
    SELECT cur.date, cur.month, 'MoM' AS performance_view,
      (cur.sessions/NULLIF(prev.sessions,0))-1 AS sessions,
      (cur.site_rr/NULLIF(prev.site_rr,0))-1 AS site_rr,
      (cur.site_conversion_rate/NULLIF(prev.site_conversion_rate,0))-1 AS site_conversion_rate,
      (cur.phone_vc/NULLIF(prev.phone_vc,0))-1 AS phone_vc,
      (cur.phone_revenue/NULLIF(prev.phone_revenue,0))-1 AS phone_revenue,
      (cur.cart_rr/NULLIF(prev.cart_rr,0))-1 AS cart_rr,
      (cur.cart_conversion_rate/NULLIF(prev.cart_conversion_rate,0))-1 AS cart_conversion_rate,
      (cur.cart_vc/NULLIF(prev.cart_vc,0))-1 AS cart_vc,
      (cur.cart_revenue/NULLIF(prev.cart_revenue,0))-1 AS cart_revenue,
      (cur.total_vc/NULLIF(prev.total_vc,0))-1 AS total_vc,
      (cur.total_revenue/NULLIF(prev.total_revenue,0))-1 AS total_revenue,
      (cur.phone_gcv_order/NULLIF(prev.phone_gcv_order,0))-1 AS phone_gcv_order,
      (cur.cart_gcv_order/NULLIF(prev.cart_gcv_order,0))-1 AS cart_gcv_order
    FROM cte_base cur
    LEFT JOIN cte_base prev ON prev.performance_view=cur.performance_view AND prev.date=add_months(cur.date,-1)
    WHERE cur.performance_view IN ('Pacing','Plan')
  ),
  cte_yoy AS (
    SELECT cur.date, cur.month, 'YoY' AS performance_view,
      (cur.sessions/NULLIF(prev.sessions,0))-1 AS sessions,
      (cur.site_rr/NULLIF(prev.site_rr,0))-1 AS site_rr,
      (cur.site_conversion_rate/NULLIF(prev.site_conversion_rate,0))-1 AS site_conversion_rate,
      (cur.phone_vc/NULLIF(prev.phone_vc,0))-1 AS phone_vc,
      (cur.phone_revenue/NULLIF(prev.phone_revenue,0))-1 AS phone_revenue,
      (cur.cart_rr/NULLIF(prev.cart_rr,0))-1 AS cart_rr,
      (cur.cart_conversion_rate/NULLIF(prev.cart_conversion_rate,0))-1 AS cart_conversion_rate,
      (cur.cart_vc/NULLIF(prev.cart_vc,0))-1 AS cart_vc,
      (cur.cart_revenue/NULLIF(prev.cart_revenue,0))-1 AS cart_revenue,
      (cur.total_vc/NULLIF(prev.total_vc,0))-1 AS total_vc,
      (cur.total_revenue/NULLIF(prev.total_revenue,0))-1 AS total_revenue,
      (cur.phone_gcv_order/NULLIF(prev.phone_gcv_order,0))-1 AS phone_gcv_order,
      (cur.cart_gcv_order/NULLIF(prev.cart_gcv_order,0))-1 AS cart_gcv_order
    FROM cte_base cur
    LEFT JOIN cte_base prev ON prev.performance_view=cur.performance_view AND prev.date=add_months(cur.date,-12)
    WHERE cur.performance_view IN ('Pacing','Plan')
  )
  SELECT * FROM cte_base
  UNION ALL SELECT * FROM cte_vs_plan
  UNION ALL SELECT * FROM cte_mom
  UNION ALL SELECT * FROM cte_yoy
)
SELECT performance_view,
  SUM(sessions) AS sessions, SUM(site_rr) AS site_rr,
  SUM(site_conversion_rate) AS site_conversion_rate,
  SUM(phone_vc) AS phone_vc, SUM(phone_revenue) AS phone_revenue,
  SUM(cart_rr) AS cart_rr, SUM(cart_conversion_rate) AS cart_conversion_rate,
  SUM(cart_vc) AS cart_vc, SUM(cart_revenue) AS cart_revenue,
  SUM(total_vc) AS total_vc, SUM(total_revenue) AS total_revenue,
  SUM(phone_gcv_order) AS phone_gcv_order, SUM(cart_gcv_order) AS cart_gcv_order
FROM q
WHERE performance_view IN ('Pacing','Plan','vs_plan','MoM','YoY')
  AND DATE_TRUNC('MONTH', month) = TO_TIMESTAMP('{month}T00:00:00.000Z')
  AND DATE_TRUNC('DAY', date) = TO_TIMESTAMP('{as_of}T00:00:00.000Z')
GROUP BY performance_view
ORDER BY performance_view ASC
"""


def _run_pacing(args_json: str) -> tuple[str, str, "pd.DataFrame | None"]:
    args = json.loads(args_json)
    month = args.get("month", _default_month_start())
    as_of = args.get("as_of_date", _default_as_of_date())
    explanation = args.get("explanation", "Running SEO finance pacing query")

    sql = _build_pacing_sql(month, as_of)
    result_str, df = execute_readonly_sql(sql, max_rows=20)
    return result_str, explanation, df


# ---------------------------------------------------------------------------
# Tool 2: GSC Summary (Current vs Prior period)
# ---------------------------------------------------------------------------

def _run_gsc_summary(args_json: str) -> tuple[str, str, "pd.DataFrame | None"]:
    args = json.loads(args_json)
    month = args.get("month", _default_month_start())
    raw_as_of = args.get("as_of_date", _default_as_of_date())
    explanation = args.get("explanation", "Running GSC summary comparison")

    as_of = _resolve_gsc_as_of(month, raw_as_of)
    prior_start = _prior_month_start(month)
    prior_date = _same_day_prior_month(as_of, month)

    sql = f"""
SELECT
  CASE WHEN date BETWEEN '{month}' AND '{as_of}' THEN 'Current'
       WHEN date BETWEEN '{prior_start}' AND '{prior_date}' THEN 'Prior' END AS period,
  SUM(clicks) AS clicks,
  SUM(impressions) AS impressions,
  ROUND(SUM(clicks)*100.0/NULLIF(SUM(impressions),0), 3) AS ctr_pct,
  ROUND(SUM(position*impressions)/NULLIF(SUM(impressions),0), 1) AS avg_rank
FROM lakehouse_production.common.gsc_search_analytics_d_1
WHERE domain IN ({_GSC_DOMAINS})
  AND (date BETWEEN '{month}' AND '{as_of}'
       OR date BETWEEN '{prior_start}' AND '{prior_date}')
GROUP BY 1
"""
    result_str, df = execute_readonly_sql(sql, max_rows=10)
    return result_str, explanation, df


# ---------------------------------------------------------------------------
# Tool 3: Page-Type Drill-Down (Apr vs Mar click comparison)
# ---------------------------------------------------------------------------

def _run_page_type_drilldown(args_json: str) -> tuple[str, str]:
    args = json.loads(args_json)
    month = args.get("month", _default_month_start())
    raw_as_of = args.get("as_of_date", _default_as_of_date())
    site_filter = args.get("site", None)
    explanation = args.get("explanation", "Running page-type drill-down")

    as_of = _resolve_gsc_as_of(month, raw_as_of)
    prior_start = _prior_month_start(month)
    prior_date = _same_day_prior_month(as_of, month)

    site_col = "gn.site," if not site_filter else ""
    site_groupby = "gn.site," if not site_filter else ""
    site_where = f"AND gn.site = '{site_filter}'" if site_filter else ""
    site_join_cols = "m.site = a.site AND" if not site_filter else ""
    site_coalesce = "COALESCE(m.site, a.site) AS site," if not site_filter else ""
    site_order = "site," if not site_filter else ""

    sql = f"""
WITH {_URL_TO_PAGE_TYPE_CTE},
gsc_norm AS (
  SELECT g.date, {_SITE_CASE} AS site,
    {_NORMALIZE_PAGE} AS landing_page,
    g.clicks, g.impressions, g.position
  FROM lakehouse_production.common.gsc_search_analytics_d_1 g
  WHERE g.date >= '{prior_start}' AND g.date <= '{as_of}'
    AND g.domain IN ({_GSC_DOMAINS})
),
tagged AS (
  SELECT
    CASE WHEN gn.date BETWEEN '{month}' AND '{as_of}' THEN 'Current'
         WHEN gn.date BETWEEN '{prior_start}' AND '{prior_date}' THEN 'Prior' END AS period,
    {site_col} COALESCE(u.landing_page_type, 'Unmatched') AS landing_page_type,
    gn.clicks, gn.impressions, gn.position
  FROM gsc_norm gn
  LEFT JOIN url_to_page_type u ON gn.landing_page = u.landing_page
  WHERE (gn.date BETWEEN '{month}' AND '{as_of}'
     OR gn.date BETWEEN '{prior_start}' AND '{prior_date}')
    {site_where}
),
prior_p AS (
  SELECT {site_groupby} landing_page_type,
    SUM(clicks) AS clicks, SUM(impressions) AS impressions,
    ROUND(SUM(clicks)*100.0/NULLIF(SUM(impressions),0),3) AS ctr_pct,
    ROUND(SUM(position*impressions)/NULLIF(SUM(impressions),0),1) AS avg_rank
  FROM tagged WHERE period = 'Prior' GROUP BY {site_groupby} landing_page_type
),
curr_p AS (
  SELECT {site_groupby} landing_page_type,
    SUM(clicks) AS clicks, SUM(impressions) AS impressions,
    ROUND(SUM(clicks)*100.0/NULLIF(SUM(impressions),0),3) AS ctr_pct,
    ROUND(SUM(position*impressions)/NULLIF(SUM(impressions),0),1) AS avg_rank
  FROM tagged WHERE period = 'Current' GROUP BY {site_groupby} landing_page_type
)
SELECT
  {site_coalesce}
  COALESCE(m.landing_page_type, a.landing_page_type) AS landing_page_type,
  COALESCE(m.clicks, 0) AS prior_clicks,
  COALESCE(a.clicks, 0) AS current_clicks,
  COALESCE(a.clicks, 0) - COALESCE(m.clicks, 0) AS click_delta,
  ROUND((COALESCE(a.clicks,0) - COALESCE(m.clicks,0)) * 100.0 / NULLIF(COALESCE(m.clicks,0), 0), 1) AS click_delta_pct,
  COALESCE(m.impressions, 0) AS prior_impressions,
  COALESCE(a.impressions, 0) AS current_impressions,
  ROUND((COALESCE(a.impressions,0) - COALESCE(m.impressions,0)) * 100.0 / NULLIF(COALESCE(m.impressions,0), 0), 1) AS impr_delta_pct,
  m.ctr_pct AS prior_ctr, a.ctr_pct AS current_ctr,
  m.avg_rank AS prior_rank, a.avg_rank AS current_rank
FROM prior_p m
FULL OUTER JOIN curr_p a ON {site_join_cols} m.landing_page_type = a.landing_page_type
ORDER BY {site_order} click_delta ASC
"""
    result_str, _ = execute_readonly_sql(sql, max_rows=100)
    return result_str, explanation


# ---------------------------------------------------------------------------
# Tool 4: Query-Level Detail for a Page Type
# ---------------------------------------------------------------------------

def _run_query_detail(args_json: str) -> tuple[str, str]:
    args = json.loads(args_json)
    page_type = args.get("landing_page_type", "Homepage")
    month = args.get("month", _default_month_start())
    raw_as_of = args.get("as_of_date", _default_as_of_date())
    site_filter = args.get("site", None)
    explanation = args.get("explanation", f"Top queries for {page_type}")

    as_of = _resolve_gsc_as_of(month, raw_as_of)
    prior_start = _prior_month_start(month)
    prior_date = _same_day_prior_month(as_of, month)
    site_where = f"AND gn.site = '{site_filter}'" if site_filter else ""

    sql = f"""
WITH {_URL_TO_PAGE_TYPE_CTE},
gsc_norm AS (
  SELECT g.date, {_SITE_CASE} AS site,
    {_NORMALIZE_PAGE} AS landing_page,
    LOWER(TRIM(g.query)) AS query,
    g.clicks, g.impressions, g.position
  FROM lakehouse_production.common.gsc_search_analytics_d_1 g
  WHERE g.date >= '{prior_start}' AND g.date <= '{as_of}'
    AND g.domain IN ({_GSC_DOMAINS})
),
tagged AS (
  SELECT
    CASE WHEN gn.date BETWEEN '{month}' AND '{as_of}' THEN 'Current'
         WHEN gn.date BETWEEN '{prior_start}' AND '{prior_date}' THEN 'Prior' END AS period,
    COALESCE(u.landing_page_type, 'Unmatched') AS landing_page_type,
    gn.query, gn.clicks, gn.impressions, gn.position
  FROM gsc_norm gn
  LEFT JOIN url_to_page_type u ON gn.landing_page = u.landing_page
  WHERE (gn.date BETWEEN '{month}' AND '{as_of}'
     OR gn.date BETWEEN '{prior_start}' AND '{prior_date}')
    {site_where}
),
curr_q AS (
  SELECT query, SUM(clicks) AS clicks, SUM(impressions) AS impressions,
    ROUND(SUM(clicks)*100.0/NULLIF(SUM(impressions),0),2) AS ctr_pct,
    ROUND(SUM(position*impressions)/NULLIF(SUM(impressions),0),1) AS avg_rank
  FROM tagged WHERE period = 'Current' AND landing_page_type = '{page_type}'
  GROUP BY query
),
prior_q AS (
  SELECT query, SUM(clicks) AS clicks, SUM(impressions) AS impressions,
    ROUND(SUM(clicks)*100.0/NULLIF(SUM(impressions),0),2) AS ctr_pct,
    ROUND(SUM(position*impressions)/NULLIF(SUM(impressions),0),1) AS avg_rank
  FROM tagged WHERE period = 'Prior' AND landing_page_type = '{page_type}'
  GROUP BY query
)
SELECT
  COALESCE(a.query, m.query) AS query,
  COALESCE(m.clicks, 0) AS prior_clicks,
  COALESCE(a.clicks, 0) AS current_clicks,
  COALESCE(a.clicks, 0) - COALESCE(m.clicks, 0) AS click_delta,
  COALESCE(m.impressions, 0) AS prior_impressions,
  COALESCE(a.impressions, 0) AS current_impressions,
  m.avg_rank AS prior_rank, a.avg_rank AS current_rank
FROM curr_q a
FULL OUTER JOIN prior_q m ON a.query = m.query
ORDER BY COALESCE(a.clicks, 0) + COALESCE(m.clicks, 0) DESC
LIMIT 40
"""
    result_str, _ = execute_readonly_sql(sql, max_rows=50)
    return result_str, explanation


# ---------------------------------------------------------------------------
# Tool 5: Monthly Rank Trend for a Site + Page Type
# ---------------------------------------------------------------------------

def _run_rank_trend(args_json: str) -> tuple[str, str]:
    args = json.loads(args_json)
    page_type = args.get("landing_page_type", "Homepage")
    site = args.get("site", "CTXP")
    explanation = args.get("explanation", f"Monthly rank trend for {site} {page_type}")

    domain_map = {
        "CTXP": "choosetexaspower.org",
        "SOE": "saveonenergy.com",
        "Choose TX": "chooseenergy.com",
        "TXER": "texaselectricrates.com",
    }
    domain = domain_map.get(site, "choosetexaspower.org")

    sql = f"""
WITH {_URL_TO_PAGE_TYPE_CTE},
gsc_monthly AS (
  SELECT
    DATE_TRUNC('month', g.date) AS month,
    COUNT(DISTINCT g.date) AS days_with_data,
    DAY(LAST_DAY(DATE_TRUNC('month', g.date))) AS days_in_month,
    SUM(g.clicks) AS clicks,
    SUM(g.impressions) AS impressions,
    ROUND(SUM(g.clicks)*100.0/NULLIF(SUM(g.impressions),0), 3) AS ctr_pct,
    ROUND(SUM(g.position*g.impressions)/NULLIF(SUM(g.impressions),0), 2) AS weighted_avg_rank
  FROM lakehouse_production.common.gsc_search_analytics_d_1 g
  LEFT JOIN url_to_page_type u
    ON {_NORMALIZE_PAGE} = u.landing_page
  WHERE g.domain = '{domain}'
    AND COALESCE(u.landing_page_type, 'Unmatched') = '{page_type}'
    AND g.date >= '2025-01-01'
  GROUP BY DATE_TRUNC('month', g.date), DAY(LAST_DAY(DATE_TRUNC('month', g.date)))
)
SELECT month, days_with_data, days_in_month, clicks, impressions, ctr_pct, weighted_avg_rank
FROM gsc_monthly ORDER BY month
"""
    result_str, _ = execute_readonly_sql(sql, max_rows=50)
    return result_str, explanation


# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_seo_pacing",
            "description": (
                "Run the SEO finance pacing query against rpt_texas_daily_pacing. "
                "Returns one row per performance_view (Pacing, Plan, vs_plan, MoM, YoY) "
                "with sessions, site_rr, site_conversion_rate, phone_vc, phone_revenue, "
                "cart_rr, cart_conversion_rate, cart_vc, cart_revenue, total_vc, total_revenue, "
                "phone_gcv_order, cart_gcv_order. Use this as the primary data source for "
                "SEO pacing status, full funnel table, and revenue waterfall decomposition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "string",
                        "description": "First day of month, e.g. '2026-04-01'. Defaults to current month.",
                    },
                    "as_of_date": {
                        "type": "string",
                        "description": "Last completed day, e.g. '2026-04-13'. Defaults to yesterday.",
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["explanation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_gsc_summary",
            "description": (
                "GSC search performance summary comparing current month-to-date vs same "
                "day-count in the prior month. Returns clicks, impressions, CTR, and "
                "weighted avg rank for each period. Use for diagnosing search visibility changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {"type": "string", "description": "First day of month, e.g. '2026-04-01'."},
                    "as_of_date": {"type": "string", "description": "Last completed day, e.g. '2026-04-13'."},
                    "explanation": {"type": "string"},
                },
                "required": ["explanation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_page_type_drilldown",
            "description": (
                "Compare GSC clicks by landing_page_type between current and prior period "
                "(same day-count). Shows click_delta, impression change, CTR, and rank per "
                "page type. Optionally filter to a single site. Use when sessions are off plan "
                "and you need to identify which page types are driving the gap."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {"type": "string"},
                    "as_of_date": {"type": "string"},
                    "site": {
                        "type": "string",
                        "description": "Optional site filter: 'CTXP', 'SOE', 'Choose TX', or 'TXER'.",
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["explanation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query_detail",
            "description": (
                "Top search queries for a specific landing_page_type, comparing current "
                "vs prior period clicks, impressions, and rank. Use to understand the keyword "
                "mix driving a page type's performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "landing_page_type": {
                        "type": "string",
                        "description": "Page type to drill into, e.g. 'Homepage', 'StateGEO', 'No_Deposit_Plans'.",
                    },
                    "month": {"type": "string"},
                    "as_of_date": {"type": "string"},
                    "site": {"type": "string", "description": "Optional site filter."},
                    "explanation": {"type": "string"},
                },
                "required": ["landing_page_type", "explanation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rank_trend",
            "description": (
                "Monthly trend of weighted avg rank, clicks, impressions, and CTR for a "
                "specific site + landing_page_type combination. Shows data from Jan 2025 onward. "
                "Use to understand long-term ranking trajectory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "description": "Site: 'CTXP', 'SOE', 'Choose TX', or 'TXER'.",
                    },
                    "landing_page_type": {
                        "type": "string",
                        "description": "Page type, e.g. 'Homepage', 'StateGEO'.",
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["site", "landing_page_type", "explanation"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_EXECUTORS = {
    "run_seo_pacing": _run_pacing,
    "run_gsc_summary": _run_gsc_summary,
    "run_page_type_drilldown": _run_page_type_drilldown,
    "run_query_detail": _run_query_detail,
    "run_rank_trend": _run_rank_trend,
}


def execute(tool_name: str, args_json: str) -> tuple[str, str, "pd.DataFrame | None"]:
    """Dispatch an SEO tool call. Returns (result_text, explanation, df_or_None)."""
    executor = _EXECUTORS.get(tool_name)
    if executor is None:
        return f"Unknown SEO tool: {tool_name}", "", None
    result = executor(args_json)
    if len(result) == 3:
        return result
    return result[0], result[1], None
