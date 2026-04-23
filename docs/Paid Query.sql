-- Daily rollup by campaign_bucket from 2025-01-01 through yesterday
WITH day_dates AS (
    SELECT
        explode(
            sequence(
                to_date('2025-01-01'), 
                date_add(current_date, -1),
                interval 1 day
            )
        ) AS day
),

-- Map campaign_id -> bucket using paid search names (scoped to the same window)
cte_ps_map AS (
    SELECT
        ps.campaign_id,
        CASE
            WHEN (
                ps.campaign_name ILIKE '%d:Supplier%'
                OR ps.campaign_name ILIKE '%Partner%'
                OR ps.campaign_name ILIKE '%d:Competitor%'
                OR ps.campaign_name ILIKE '%d:CustomerService%'
                OR ps.campaign_name ILIKE '%d:Utility%'
            ) THEN 'Supplier'
            WHEN ps.campaign_name ILIKE "%4000148%" THEN 'Rates'
            WHEN ps.campaign_name ILIKE '%d:Aggregator%' THEN 'Aggregator'
            WHEN ps.campaign_name ILIKE '%pMax%' THEN 'PMax'
            WHEN ps.campaign_name ILIKE '%d:Companies%' THEN 'Companies'
            WHEN ps.campaign_name ILIKE '%d:PriceSens%' THEN 'Price Sensitive'
            WHEN ps.campaign_name ILIKE '%d:Geo%' THEN 'Geo'
            WHEN (
                ps.campaign_name ILIKE '%d:Generic%'
                OR ps.campaign_name ILIKE '%d:LowPerformance%'
                OR ps.campaign_name ILIKE '%d:TrueBroad%'
            ) THEN 'Generic'
            WHEN ps.campaign_name ILIKE '%d:Spanish%' THEN 'Spanish'
            WHEN ps.campaign_name ILIKE '%Brand%' THEN 'Brand'
            WHEN ps.campaign_name ILIKE "%4000150%" THEN 'NoDeposit'
            ELSE 'Other'
        END AS campaign_bucket
    FROM energy_prod.energy.paidsearch_campaign ps
    JOIN day_dates dd
        ON ps.rpt_date = dd.day
    WHERE ps.campaign_name ILIKE '%G:TX%'
      AND ps.campaign_name NOT ILIKE '%Display%'
      AND (
            ps.campaign_name ILIKE '%a:CTXP%'
         OR ps.campaign_name ILIKE '%a:SOE%'
         OR ps.campaign_name ILIKE '%a:TXER%'
         OR ps.campaign_name ILIKE '%a:CHOO%'
      )
    GROUP BY
        ps.campaign_id,
        CASE
            WHEN (
                ps.campaign_name ILIKE '%d:Supplier%'
                OR ps.campaign_name ILIKE '%Partner%'
                OR ps.campaign_name ILIKE '%d:Competitor%'
                OR ps.campaign_name ILIKE '%d:CustomerService%'
                OR ps.campaign_name ILIKE '%d:Utility%'
            ) THEN 'Supplier'
            WHEN ps.campaign_name ILIKE "%4000148%" THEN 'Rates'
            WHEN ps.campaign_name ILIKE '%d:Aggregator%' THEN 'Aggregator'
            WHEN ps.campaign_name ILIKE '%pMax%' THEN 'PMax'
            WHEN ps.campaign_name ILIKE '%d:Companies%' THEN 'Companies'
            WHEN ps.campaign_name ILIKE '%d:PriceSens%' THEN 'Price Sensitive'
            WHEN ps.campaign_name ILIKE '%d:Geo%' THEN 'Geo'
            WHEN (
                ps.campaign_name ILIKE '%d:Generic%'
                OR ps.campaign_name ILIKE '%d:LowPerformance%'
                OR ps.campaign_name ILIKE '%d:TrueBroad%'
            ) THEN 'Generic'
            WHEN ps.campaign_name ILIKE '%d:Spanish%' THEN 'Spanish'
            WHEN ps.campaign_name ILIKE '%Brand%' THEN 'Brand'
            WHEN ps.campaign_name ILIKE "%4000150%" THEN 'NoDeposit'
            ELSE 'Other'
        END
),

/* ===== FACTS (scoped to same window) ===== */

cte_paidsearch AS (
    SELECT
        ps.rpt_date AS day,
        ps.campaign_id,
        SUM(ps.impressions) AS impressioncount,
        SUM(ps.clicks) AS clickcount,
        SUM(ps.spend) AS cost
    FROM energy_prod.energy.paidsearch_campaign ps
    JOIN day_dates dd
        ON ps.rpt_date = dd.day
    WHERE ps.campaign_name ILIKE '%G:TX%'
      AND ps.campaign_name NOT ILIKE '%Display%'
      AND (
            ps.campaign_name ILIKE '%a:CTXP%'
         OR ps.campaign_name ILIKE '%a:SOE%'
         OR ps.campaign_name ILIKE '%a:TXER%'
         OR ps.campaign_name ILIKE '%a:CHOO%'
      )
    GROUP BY
        ps.rpt_date,
        ps.campaign_id
),

cte_sessions AS (
    SELECT
        DATE(s.session_start_ts_est) AS day,
        s.cmpid AS campaign_id,
        COUNT(DISTINCT s.webcontext_session_id) AS sessions
    FROM energy_prod.energy.v_sessions s
    JOIN day_dates dd
        ON DATE(s.session_start_ts_est) = dd.day
    WHERE s.bot_ip_ind = 0
      AND s.is_bot = false
      AND (s.is_internal_ip = FALSE OR s.is_internal_ip IS NULL)
      AND COALESCE(s.company_id, 25) = 25
      AND s.first_page_url NOT ILIKE '%solar-energy%'
      AND s.first_page_url NOT ILIKE '%resources%'
      AND (s.non_texas_ind = 0 OR s.campaign ILIKE '%a:SOE%')
      AND (s.category IS NULL OR s.category != 'Non-Texas')
      AND (
            LOWER(s.region) = 'tx'
         OR s.first_page_url ILIKE '%texas%'
         OR s.site_name NOT ILIKE '%chooseenergy%'
      )
    GROUP BY
        DATE(s.session_start_ts_est),
        s.cmpid
),

cte_carts AS (
    SELECT
        DATE(c.cartsession_start_ts_est) AS day,
        c.cmpid AS campaign_id,
        COUNT(DISTINCT c.cartsession_id) AS cart_starts
    FROM energy_prod.energy.v_carts c
    JOIN day_dates dd
        ON DATE(c.cartsession_start_ts_est) = dd.day
    WHERE c.bot_ip_ind = 0
    GROUP BY
        DATE(c.cartsession_start_ts_est),
        c.cmpid
),

cte_calls AS (
    SELECT
        DATE(cl.call_ts_est) AS day,
        cl.campaign_id,
        COUNT(DISTINCT CASE WHEN cl.ib_gross_ind = 1 THEN cl.call_id END) AS gross_calls,
        COUNT(DISTINCT CASE WHEN cl.ib_queue_ind = 1 THEN cl.call_id END) AS queue_calls,
        COUNT(DISTINCT CASE WHEN cl.ib_queue_ind > 0 AND UPPER(cl.call_lease) LIKE '%FUSE-PERMA-GRID%' THEN cl.call_id END) AS queue_calls_grid,
        COUNT(DISTINCT CASE WHEN cl.ib_queue_ind > 0 AND UPPER(cl.call_lease) LIKE '%FUSE-PERMA-BANNER%' THEN cl.call_id END) AS queue_calls_homepage,
        COUNT(DISTINCT CASE WHEN cl.ib_queue_ind > 0 AND UPPER(cl.call_lease) NOT LIKE '%FUSE-PERMA-GRID%' AND UPPER(cl.call_lease) NOT LIKE '%FUSE-PERMA-BANNER%' AND cl.permalease_call_ind != 1 THEN cl.call_id END) AS queue_calls_other,
        COUNT(DISTINCT CASE WHEN cl.ibs_net_ind = 1 THEN cl.call_id END) AS net_calls,
        COUNT(DISTINCT CASE WHEN cl.ib_gross_ind = 1 AND cl.permalease_call_ind = 1 THEN cl.call_id END) AS gross_serp,
        COUNT(DISTINCT CASE WHEN cl.ib_queue_ind = 1 AND cl.permalease_call_ind = 1 THEN cl.call_id END) AS queue_serp,
        COUNT(DISTINCT CASE WHEN cl.ibs_net_ind = 1 AND cl.permalease_call_ind = 1 THEN cl.call_id END) AS net_serp,
        SUM(cl.salescentercost) AS scc
    FROM energy_prod.energy.v_calls cl
    JOIN day_dates dd
        ON DATE(cl.call_ts_est) = dd.day
    GROUP BY
        DATE(cl.call_ts_est),
        cl.campaign_id
),

cte_orders AS (
    SELECT
        DATE(o.order_ts_est) AS day,
        o.cmpid AS campaign_id,
        COUNT(DISTINCT CASE WHEN o.order_type = 'Phone' THEN o.order_id END) AS phone_orders,
        COUNT(DISTINCT CASE WHEN o.order_type = 'Cart' THEN o.order_id END) AS cart_orders2,
        COUNT(DISTINCT CASE WHEN o.order_type = 'Phone' AND o.permalease_call_ind = 1 THEN o.order_id END) AS serp_orders,
        SUM(gcv.gcv_new_v2) AS est_rev
    FROM energy_prod.energy.v_orders o
    LEFT JOIN energy_prod.energy.v_orders_gcv gcv
        ON o.order_id = gcv.order_id
    JOIN day_dates dd
        ON DATE(o.order_ts_est) = dd.day
    WHERE (o.tenant_id != 'src_1ax2zVVJJ2cB9aBFEnVcRPog5Pe' OR o.tenant_id IS NULL)
    GROUP BY
        DATE(o.order_ts_est),
        o.cmpid
)

-- FINAL daily rollup by bucket
SELECT
    dd.day AS day,
    m.campaign_bucket AS campaign_bucket,
    TO_CHAR(DATE_TRUNC(:grain, dd.day), 'yyyy-MM-dd') AS date_grain,
    DAYNAME(dd.day) AS day_of_week,
    DAY(dd.day) AS day_of_month,
    COALESCE(SUM(ps.impressioncount), 0) AS impressioncount,
    COALESCE(SUM(ps.clickcount), 0) AS clickcount,
    COALESCE(SUM(ps.cost), 0) AS cost,
    COALESCE(SUM(s.sessions), 0) AS sessions,
    COALESCE(SUM(ct.cart_starts), 0) AS cart_starts,
    COALESCE(SUM(cl.gross_calls), 0) AS gross_calls,
    COALESCE(SUM(cl.queue_calls), 0) AS queue_calls,
    
    COALESCE(SUM(cl.net_calls), 0) AS net_calls,
    COALESCE(SUM(cl.gross_serp), 0) AS gross_serp,
    COALESCE(SUM(cl.queue_serp), 0) AS queue_serp,
    COALESCE(SUM(cl.net_serp), 0) AS net_serp,
    COALESCE(SUM(cl.scc), 0) AS scc,
    COALESCE(SUM(o.phone_orders), 0) AS phone_orders,
    COALESCE(SUM(o.cart_orders2), 0) AS cart_orders2,
    COALESCE(SUM(o.serp_orders), 0) AS serp_orders,
    COALESCE(SUM(o.phone_orders), 0) - COALESCE(SUM(o.serp_orders), 0) AS site_phone_orders, 
    COALESCE(SUM(o.est_rev), 0) AS est_rev,
    COALESCE(SUM(cl.queue_calls), 0) - COALESCE(SUM(cl.queue_serp), 0) AS site_queue_calls,
    COALESCE(SUM(cl.queue_calls_grid), 0) AS queue_calls_grid,
    COALESCE(SUM(cl.queue_calls_homepage), 0) AS queue_calls_homepage,
    COALESCE(SUM(cl.queue_calls_other), 0) AS queue_calls_other

FROM day_dates dd
JOIN cte_ps_map m
    ON 1 = 1
LEFT JOIN cte_paidsearch ps
    ON ps.day = dd.day
   AND ps.campaign_id = m.campaign_id
LEFT JOIN cte_sessions s
    ON s.day = dd.day
   AND s.campaign_id = m.campaign_id
LEFT JOIN cte_carts ct
    ON ct.day = dd.day
   AND ct.campaign_id = m.campaign_id
LEFT JOIN cte_calls cl
    ON cl.day = dd.day
   AND cl.campaign_id = m.campaign_id
LEFT JOIN cte_orders o
    ON o.day = dd.day
   AND o.campaign_id = m.campaign_id
GROUP BY
    dd.day,
    m.campaign_bucket,
    TO_CHAR(DATE_TRUNC(:grain, dd.day), 'yyyy-MM-dd'),
    DAYNAME(dd.day),
    DAY(dd.day)
ORDER BY
    day DESC,
    campaign_bucket;
