# PRD: Energy Marketplace Performance Reporting Tool v2

## Problem Statement

The current Streamlit app ("Funnel KPI Driver Analysis") is a strong proof-of-concept for session-level decomposition analysis, but it has outgrown its original scope. It serves a single scrolling page with seven sections, mixing monthly pacing, weekly KPI decomposition, initiative impact, LLM narrative, and a chat agent into one flow. As usage matures, the team needs:

1. **A generalized reporting tool** where any analyst can select a KPI, marketing channel(s), and date range and immediately see pacing, period-over-period, and waterfall views — without needing to understand the underlying data model.
2. **Deep-dive tabs for focus channels** (Organic/SEO and Paid Search today) that surface channel-specific diagnostics not relevant to other channels — impression/click/ranking analysis for SEO, auction/CPC/impression-share analysis for Paid Search.
3. **An AI layer that truly understands the business** — not just the data schema, but what each KPI means, how the funnel works, what initiatives do, and what seasonality patterns exist — so that narratives and chat responses are genuinely useful rather than generic.

---

## Non-Negotiables


| #   | Requirement                                                                             | Why                                                                                             |
| --- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1   | **Any KPI × Channel × Date combination** via sidebar selectors                          | Analysts must be able to explore any slice without code changes                                 |
| 2   | **Month Pacing & Period KPIs table** (Pacing, Plan, vs Plan, Current Period, PoP, P4PA) | This is the first thing management looks at — source-of-truth from finance data                 |
| 3   | **Period-over-Period (PoP) decomposition** (mix-shift + rate-effect)                    | Core analytical value — explains *why* a KPI moved                                              |
| 4   | **vs Past 4-Period Average (P4WA)** comparison                                          | Smooths noise; management uses this to distinguish signal from random week-to-week variation    |
| 5   | **Revenue Waterfall** (sequential substitution: Plan → Actual)                          | Translates rate changes into dollar impact; most important view for revenue conversations       |
| 6   | **Channel Deep-Dive tabs** for Organic (SEO) and Paid Search                            | These channels have unique upstream data (GSC, Google Ads) that require specialized diagnostics |
| 7   | **Business-aware AI** — the tool must embed full business context                       | LLM narratives and chat must explain causality, not restate numbers                             |


---

## Architecture

### Tab Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  Sidebar                                                        │
│  ┌──────────────────┐                                           │
│  │ KPI Selector      │  (ZLUR, Cart RR, SSN Submit, Cart Conv, │
│  │                   │   Cart VC, Phone RR, Phone VC, VC)       │
│  ├──────────────────┤                                           │
│  │ Channel Filter    │  (multi-select, defaults: Paid Search,   │
│  │                   │   Direct, Organic, pMax)                  │
│  ├──────────────────┤                                           │
│  │ Website Filter    │  (CTXP, SOE, Choose TX, TXER, all)       │
│  ├──────────────────┤                                           │
│  │ Time Mode         │  (WoW, WTD Tue, WTD Fri, MoM, Custom)   │
│  ├──────────────────┤                                           │
│  │ Date Inputs       │  (auto or manual)                        │
│  └──────────────────┘                                           │
│                                                                  │
│  ┌─────────┬────────────┬────────────┬──────────────┐           │
│  │Overview │ Deep Dive: │ Deep Dive: │ Ask the      │           │
│  │         │ Organic    │ Paid Search│ Analyst      │           │
│  └─────────┴────────────┴────────────┴──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Tab 1: Overview (All Channels)

This is the primary view. It works for any KPI × Channel × Date combination.

**Section 1: Performance Summary (Finance Source of Truth)**

The existing `build_funnel_summary` → `render_summary_html` table. Columns:


| Given marketing channels Funnel | {Month} Pacing | Plan Actual | V. Plan | {Period Label} | WoW | P4WA |
| ------------------------------- | -------------- | ----------- | ------- | -------------- | --- | ---- |


- Source: `energy_prod.energy.rpt_texas_daily_pacing` (Pacing/Plan) + `finance_query` (actuals)
- Rows: LP Sessions, Phone RR, Phone VC, ZLUR, G2C, Cart RR, Cart Conversion, Cart VC, VC, Cart Revenue, Rev/Session, Rev/Order
- Conditional formatting: green (>+2.5%), red (<-2.5%), yellow (flat)
- **Enhancement:** Add a sparkline or mini trend column showing daily pacing trajectory for the current month

**Section 2: Selected KPI Summary**

Metric cards: Current rate, Prior rate, Current volume (num/den), Prior volume.

Full funnel expander showing all KPIs in a DataFrame.

**Section 3: What Drove the Change (PoP Decomposition)**

Horizontal stacked bar chart (Plotly): top N drivers across all dimensions, split into mix effect (lighter) and rate effect (darker). Each bar labeled with total pp contribution.

- Dimensions: Site, Channel, Mover/Switcher, Device, Landing Page Type (Organic only), Entry Partner
- Initiative WoW change bars appended when material
- Expander: full driver detail table (Segment, Current Share, Prior Share, Current Rate, Prior Rate, Mix Effect, Rate Effect, Total)

**Section 4: Revenue Waterfall**

A new section (does not exist in v1). Two sub-sections:

*4a. Pacing vs Plan Revenue Waterfall*

- Uses the sequential substitution method from the SEO reporting agent rule
- Phone path: Sessions → Site RR → Site Conversion → Phone GCV/Order
- Cart path: Sessions → Cart RR → Cart Conversion → Cart GCV/Order
- Plotly waterfall chart: Plan Revenue at left, each factor's dollar impact as increase/decrease bars, Pacing Revenue at right
- Narrative callout: "Sessions are the largest drag (-$X). Cart GCV/Order is a tailwind (+$Y)."

*4b. WoW Revenue Bridge (optional — when time mode is WoW/WTD)*

- Same waterfall but comparing current week to prior week
- Uses finance actuals rather than plan

**Section 5: Dimension Detail**

Tabs per dimension (Site, Channel, Mover/Switcher, Device, Landing Page, Entry Partner). Each tab shows the full decomposition table.

**Section 6: Initiative Impact**

Existing model vs holdout analysis. Metric cards (total initiative contribution, prior, change). Bar chart. Period-over-period comparison table.

**Section 7: Cart Conversion Decomposition**

Only shown when KPI is Cart Conversion or VC. SSN Submit Rate × Conversion After Credit breakdown.

**Section 8: AI Weekly Summary**

GPT-4o generated narrative. Same as current but with enhanced business context (see "Business Knowledge" section below).

**Section 9: Ask the Analyst**

Chat interface with tool calling (pandas queries on loaded DataFrames, SQL against Databricks). Same as current but with richer system prompt.

---

### Tab 2: Deep Dive — Organic (SEO)

Auto-filters to `marketing_channel = 'Organic'`. Sidebar channel filter is overridden.

This tab reuses the analysis built in `notebooks/seo_landing_page_type_analysis.ipynb` and the protocol from `.cursor/rules/seo-reporting-agent.mdc`.

**Section 1: SEO Pacing Summary**

Finance pacing table filtered to SEO only. Same format as Overview Section 1 but single-channel.

**Section 2: SEO Revenue Waterfall**

Waterfall chart for Organic only: Plan Revenue → Sessions → Site RR → Site Conversion → Phone GCV/Order → Cart RR → Cart Conversion → Cart GCV/Order → Pacing Revenue.

**Section 3: Top-of-Funnel Visibility (GSC)**

Four-panel monthly trend chart (Plotly subplots):

- Impressions (bar)
- Clicks (bar)
- CTR % (line)
- Weighted Average Rank (line, inverted y-axis so lower = better)

Data source: `lakehouse_production.common.gsc_search_analytics_d_5` (total organic metrics — matches GSC dashboard exactly)

Domains: `choosetexaspower.org`, `saveonenergy.com`, `chooseenergy.com`, `texaselectricrates.com`

Site filter dropdown: All, CTXP, SOE, Choose TX, TXER (maps domain → website name)

**Section 4: Click Decomposition**

Compare current vs prior period (same day count). Report:

- Total click change
- Impression effect: `(Δ impressions) × prior CTR`
- CTR effect: `(Δ CTR) × prior impressions`
- Diagnostic decision tree output (rankings changed? impressions changed? CTR changed?)

**Section 5: Performance by Landing Page Type**

Table and bar charts breaking down GSC clicks, impressions, CTR, and weighted avg rank by `landing_page_type` × `site`. Highlights the top 5 page types driving click losses/gains.

Uses the `url_to_page_type` CTE from the notebook:

```sql
SELECT landing_page, landing_page_type FROM (
  SELECT
    RTRIM('/', LOWER(first_page_url)) AS landing_page,
    landing_page_type,
    ROW_NUMBER() OVER (
      PARTITION BY RTRIM('/', LOWER(first_page_url))
      ORDER BY SUM(sessions) DESC
    ) AS rn
  FROM energy_prod.data_science.mp_session_level_query
  WHERE _date >= '2025-01-01' AND landing_page_type IS NOT NULL
  GROUP BY RTRIM('/', LOWER(first_page_url)), landing_page_type
) WHERE rn = 1
```

**Section 6: Top Queries by Page Type**

Table showing top 10 GSC queries per landing_page_type, ranked by clicks. Filterable by site.

**Section 7: Session Funnel by Page Type**

Sessions, zip entries, carts, orders, VC% broken down by `landing_page_type` from `mp_session_level_query` (filtered to Organic).

---

### Tab 3: Deep Dive — Paid Search

Auto-filters to `marketing_channel IN ('Paid Search', 'pMax')`. Sidebar channel filter is overridden.

Data source: `docs/Paid Query.sql` against `energy_prod.energy.paidsearch_campaign` + `v_sessions` + `v_carts` + `v_calls` + `v_orders`, joined by `campaign_id`. The `:grain` bind variable controls time aggregation (day/week/month).

**Section 1: Paid Pacing Summary**

Finance pacing table filtered to Paid Search + pMax. Source: `rpt_texas_daily_pacing`.

**Section 2: Paid Search Revenue Waterfall (vs Plan)**

Same sequential substitution waterfall methodology applied to Paid Search funnel.

**Section 3: Campaign Bucket Performance Table**

Central table powered by the Paid Query. Columns:


| Campaign Bucket | Impressions | Clicks | CTR | Cost | CPC | Sessions | Cart Entries | Orders | Revenue | VC  |     |
| --------------- | ----------- | ------ | --- | ---- | --- | -------- | ------------ | ------ | ------- | --- | --- |


- Period selector: current period vs prior period (same as sidebar time mode)
- Delta columns: show % change for each metric
- Sortable by any column
- Highlight: top 3 movers (largest absolute change in revenue)

Campaign buckets: Supplier, Aggregator, Brand, Companies, Generic, Geo, Price Sensitive, Spanish, Rates, NoDeposit, PMax, Other. Derived from campaign naming convention in `paidsearch_campaign.campaign_name` (see Campaign Bucket Mapping in Data Architecture section).

**Section 4: Campaign Bucket Waterfall**

A waterfall chart showing which campaign buckets drove the period-over-period KPI change. The logic here should be the same as the other driver waterfall - so a counterfactual analysis on what was campaign mix and campaign rate driven. 

- X-axis: campaign buckets, ordered by impact magnitude
- Y-axis: dollar change in revenue (`bucket_current_rev - bucket_prior_rev`)
- Color: green for growth, red for decline
- Label: dollar amount + % of total change

**Section 6: Campaign Bucket Drill-Down**

When a user selects a campaign bucket, show:

- Daily trend of impressions, clicks, cost, sessions for that bucket
- Full funnel: Sessions → Carts → Orders → Revenue
- Phone funnel: Gross Calls → Queue Calls (grid / homepage / other) → Net Calls → Phone Orders (including SERP breakout via `queue_serp`, `net_serp`)
- Efficiency: CPC, Cost per Order, ROAS trended

---

### Tab 4: Ask the Analyst

Full-page chat interface (same as current Section 7 but promoted to its own tab for more space). Includes:

- All loaded DataFrames available for pandas queries
- Databricks SQL access for ad-hoc exploration
- Full business context in the system prompt
- Tool execution results displayed inline

---

## Business Knowledge Layer

The AI must understand the business deeply. This is encoded in three places:

### 1. Cursor Rule (`.cursor/rules/seo-reporting-agent.mdc`)

Already built. Governs behavior when the agent is asked about SEO pacing in Cursor.

### 2. LLM System Prompt (in `narrative.py` and `analyst_tools.py`)

Expand the existing system prompt to include:

**Business Model:**

- Energy marketplace connecting Texas consumers with electricity providers
- Four sites: ChooseTexasPower.org (CTXP, largest), SaveOnEnergy.com (SOE), ChooseEnergy.com (Choose TX), TexasElectricRates.com (TXER)
- Revenue model: commission per enrollment from electricity providers

**Full Funnel (from `full_funnel.docx`):**

- Session → ZIP Entry (ZLUR) → Cart Entry (Cart RR) → SSN/Credit Check → Order (Cart Conversion)
- Parallel phone path: Session → Site Queue Call (Site RR) → Phone Order (Site Conversion)
- VC = (Cart Orders + Phone Orders) / Sessions (excludes SERP orders)
- Revenue = Phone Revenue + Cart Revenue; GCV/Order is the average commission per order

**KPI Definitions (from `config.py`):**

- ZLUR: zip_entry / session — intent proxy, measures product engagement
- Cart RR: has_cart / session — measures cart reach
- SSN Submit Rate: cart_ssn_done / has_cart — pre-credit friction
- Conversion After Credit: cart_order / cart_ssn_done — post-credit conversion
- Cart Conversion: cart_order / has_cart — end-to-end cart conversion
- Cart VC: cart_order / session — cart path visit conversion
- Phone RR: queue_call / session — phone reach rate
- Phone VC: phone_order / session — phone path visit conversion
- VC: total_order / session — total visit conversion

**Initiatives:**

- LP JO (Landing Page Journey Optimization): ML-based personalization of landing page content. Model vs Holdout design.
- Grid JO (Grid Journey Optimization): ML-based personalization of plan grid ordering. Model vs Holdout design.
- FMP (Find My Plan): Guided plan-matching flow via LP agent 3577 ("intuitive_explore"). Shares LP holdout as control. Can overlap with Grid JO.

**Seasonality:**

- Texas electricity market is highly seasonal
- Summer (June-August): Peak demand, highest sessions and orders (heat drives plan shopping)
- Winter (December-February): Secondary peak (heating costs)
- Spring (March-May) and Fall (September-November): Shoulder seasons, lower demand
- Weekly pattern: Monday-Tuesday highest, weekend lowest

**Channel Characteristics:**

- Organic/SEO: Largest volume channel. Session count is driven by Google rankings, impressions, and CTR. Changes are slow (ranking improvements take weeks). Sensitive to algorithm updates and SERP feature changes (AI Overviews).
- Paid Search: Second largest. Controllable via bids and budgets. CPC-sensitive. Includes both traditional search and pMax.
- Direct: Brand recognition traffic. Tends to convert well. Not directly controllable.
- pMax: Google's automated campaign type. Black box auction dynamics.

**Analytical Concepts:**

- Mix effect: segment share changed (more/less traffic from that segment)
- Rate effect: segment's conversion rate changed
- Sequential substitution waterfall: swap Plan → Actual one factor at a time to attribute dollar impact
- Pacing: projected full-month value from MTD actuals using pacing weights (day-of-month × seasonality)
- P4WA: Past 4-period weighted average, used to smooth noise

### 3. Business Context Documents

Store reference documents in `docs/business_context/`:

- `full_funnel.docx` — KPI definitions and funnel logic (already exists)
- `operational_overview.docx` — business operations and channel strategy (create if missing)
- `personalization_initiatives.docx` — LP JO, Grid JO, FMP design and holdout methodology (create if missing)

These are referenced by the system prompt as grounding material.

---

## Data Architecture & Reconciliation

This is the most critical section for making the AI agent effective. The agent must understand not just *what* tables exist but *how they relate*, *where they disagree*, and *which to trust* for each use case.

### The Three Tiers of Data

```
Tier 1: FINANCE (source of truth for performance reporting)
  ├── finance_query (file) → daily actuals from v_sessions, v_carts, v_orders, v_calls
  ├── rpt_texas_daily_pacing → paced month projections and official Plan
  └── Trust level: HIGHEST — use for all pacing, plan comparisons, revenue numbers

Tier 2: SESSION-LEVEL (source of truth for decomposition / initiative analysis)
  ├── session_level_query (file) → one row per session with full funnel flags
  ├── mp_session_level_query (materialized view on Databricks)
  └── Trust level: DIRECTIONAL — great for WHY questions, may not tie exactly to Tier 1

Tier 3: CHANNEL-SPECIFIC UPSTREAM (source of truth for channel diagnostics)
  ├── GSC (all-in organic): lakehouse_production.common.gsc_search_analytics_d_5
  │     └── Use for total organic clicks/impressions — matches GSC dashboard 100%
  ├── GSC (page-level): lakehouse_production.common.gsc_search_analytics_d_3
  │     └── Use for page-type and landing-page drill-downs (has page column)
  ├── SEO Clarity: lakehouse_production.common.seo_fact_clarity_keywords_rankings_json
  │     └── Supplement to d_3 for keyword-level ranking tracking
  ├── Paid Search: energy_prod.energy.paidsearch_campaign (Paid)
  └── Trust level: HIGHEST for their specific domain, but not directly joinable to Tier 1/2
```

### Why Finance ≠ Session-Level (Known Gaps)

**Sessions should match 1:1** between `finance_query` and `session_level_query`. Both use the same source table (`energy_prod.energy.v_sessions`) with the same bot/Texas filters. If they diverge, the likely culprit is:

- Date boundary handling (`session_start_date_est` vs `rpt_date`)
- The `is_internal_ip` filter (finance uses `= FALSE`, session-level uses `= FALSE OR IS NULL`)

**Cart and order metrics WILL be slightly off** because:

1. **Channel assignment timing**: Finance assigns channel to carts/orders based on the cart/call session's own traffic source. Session-level assigns channel based on the *web session* that started the cart. For calls, the finance query does outbound order reclassification (30-day lookback) which the session-level query does not.
2. **Cart counting**: Finance counts `COUNT(DISTINCT websession_id)` from `v_carts`. Session-level flags `has_cart` based on a left join from `v_carts` to `base_sessions` — these are equivalent but the channel assignment on each side can differ.
3. **Order counting**: Finance uses `v_orders` joined to `v_calls` with outbound reclass CTE. Session-level uses cart_orders from `v_orders` joined through `v_carts`, and phone_orders from `v_calls` → `v_orders`. The phone path reclass logic differs.
4. **GCV/Revenue**: Finance uses `v_orders_gcv.gcv_new_v2`. Session-level uses the same but joins through different paths.

**Rule for the AI**: Always present finance numbers as the headline. Use session-level numbers only in the decomposition/driver sections. If a user asks "why don't these match?", explain the channel attribution timing difference.

### Underlying Databricks Tables


| Table                                                                   | Schema                    | What It Is                                                                                                                                                                                                  |
| ----------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `energy_prod.energy.v_sessions`                                         | Session-level fact        | One row per web session. Contains traffic source/channel, device, site, region, bot flags, `transactional_ind`, `cmpid` (campaign ID for paid). Both finance and session-level queries start here.          |
| `energy_prod.energy.v_carts`                                            | Cart-level fact           | One row per cart session. Joined to sessions via `websession_id`. Contains `cartsession_id`, traffic source, bot flags, partner info.                                                                       |
| `energy_prod.energy.v_orders`                                           | Order-level fact          | One row per order. Contains `order_type` (Cart/Phone), `cmpid`, GCV via join to `v_orders_gcv`. Tenant filter: exclude `src_1ax2zVVJJ2cB9aBFEnVcRPog5Pe`.                                                   |
| `energy_prod.energy.v_orders_gcv`                                       | Order revenue             | `gcv_new_v2` is the estimated commission per order. Join on `order_id`.                                                                                                                                     |
| `energy_prod.energy.v_calls`                                            | Call-level fact           | One row per call. Contains `web_session_id`, `ib_queue_ind`, `ibs_net_ind`, `permalease_call_ind`, `call_lease`, `dnis`. SERP calls identified by `permalease_call_ind = 1`.                                |
| `energy_prod.energy.paidsearch_campaign`                                | Google Ads campaign data  | Daily impressions, clicks, spend by `campaign_id`. Campaign naming convention encodes bucket (`d:Supplier`, `d:Generic`, etc.) and site (`a:CTXP`, `a:SOE`, etc.).                                          |
| `energy_prod.energy.rpt_texas_daily_pacing`                             | Finance pacing rollup     | Pre-aggregated daily metrics by channel × performance_view (Pacing, Final/Plan, Internal Plan). The "official" numbers for management reporting.                                                            |
| `energy_prod.fivetran_finance.pacing_weights`                           | Pacing weight table       | Day-level weights for projecting MTD to full-month. Used by `finance_query`.                                                                                                                                |
| `lakehouse_production.common.gsc_search_analytics_d_5`                  | GSC — site-level totals   | Daily clicks, impressions, CTR, position by domain and device. **No page or query columns.** Matches the GSC dashboard 100%. Use for all-in organic performance (total clicks, impressions, CTR trends).    |
| `lakehouse_production.common.gsc_search_analytics_d_3`                  | GSC — page-level          | Daily clicks, impressions, position, CTR by page, domain, device. **Has page column but no query column.** Use for page-type drill-downs and landing page analysis.                                         |
| `lakehouse_production.common.gsc_search_analytics_d_1`                  | GSC — query-level         | Daily clicks, impressions, position by query, page, domain, device. Most granular but totals will NOT match the GSC dashboard due to anonymized/long-tail query aggregation. Use for query bucketing only.   |
| `lakehouse_production.common.seo_fact_clarity_keywords_rankings_json`   | SEO Clarity rankings      | Keyword-level ranking tracking: `organic_results_web_rank`, `organic_results_true_rank`, `avg_search_volume`, `device`, `keyword_tracked`, `keyword_tags`. Supplements d_3 for keyword-level ranking distribution analysis. |
| `lakehouse_production.energy.dapi_decisions`                            | DAPI experiment decisions | Records which sessions got LP JO, Grid JO, or FMP treatment. `agentId = '3577'` for LP, `'3378'`/`'3401'` for Grid.                                                                                         |
| `energy_prod.data_science.dapi_decision_explore_exploit_classification` | Explore/exploit labels    | Classifies DAPI decisions as `explore`, `exploit`, or `intuitive_explore` (FMP).                                                                                                                            |


### GSC Table Selection Guide

The GSC tables in `lakehouse_production.common` represent different aggregation levels from the Google Search Console API. Each level trades granularity for accuracy — the more dimensions you request, the more Google anonymizes/drops long-tail data.

```
d_5 (site-level)     ← MATCHES GSC DASHBOARD 100%
 │  Columns: date, domain, device, clicks, impressions, ctr, position
 │  No page, no query
 │  Use: total organic click/impression trends, CTR, avg rank
 │
d_3 (page-level)     ← USE FOR PAGE-TYPE ANALYSIS
 │  Columns: date, domain, device, page, clicks, impressions, ctr, position
 │  No query
 │  Use: page-type drill-downs, landing page performance, site × page_type breakdowns
 │  Note: SUM(clicks) will be slightly less than d_5 due to anonymized pages
 │
d_1 (query-level)    ← MOST GRANULAR, LEAST COMPLETE
    Columns: date, domain, device, page, query, clicks, impressions, ctr, position
    Use: query bucketing (seo_query_bucket_map) only
    Note: SUM(clicks) will NOT match GSC dashboard — Google drops anonymized queries
```

**Rule for the AI:**
- For "how are organic clicks/impressions trending?" → use **d_5**
- For "which page types gained/lost clicks?" → use **d_3**
- For "what queries drive this page type?" → use **d_1** (accepting the undercount)
- For keyword ranking distributions → use **SEO Clarity** as a supplement to d_3

### Query Inventory (What Already Exists)


| Query File                             | Location     | Parameterized?                                      | Grain                            | Notes                                                                                                                                                                                     |
| -------------------------------------- | ------------ | --------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `finance_query`                        | project root | Dates via CTE                                       | Date × Channel                   | Source of truth for actuals. ~600 lines. Includes outbound order reclass, SERP logic, pacing weights.                                                                                     |
| `session_level_query`                  | project root | Dates via CTE                                       | Session                          | Source of truth for decomposition. ~745 lines. Includes cart funnel, phone funnel, initiative flags, credit quality.                                                                      |
| `Finance SEO Pacing Query`             | `notebooks/` | Bind variable `:marketing_channel`, hardcoded dates | Date × Channel × Perf View       | The pacing/plan query that powers the funnel summary. Uses `rpt_texas_daily_pacing`.                                                                                                      |
| `Paid Query.sql`                       | `docs/`      | Bind variable `:grain`                              | Date × Campaign Bucket           | Paid search campaign-level rollup. Joins `paidsearch_campaign`, `v_sessions`, `v_carts`, `v_calls`, `v_orders` by `campaign_id`. Campaign bucket derived from campaign naming convention. |
| `seo_landing_page_type_analysis.ipynb` | `notebooks/` | Hardcoded dates                                     | Month × Site × Landing Page Type | GSC + session metrics by landing page type. Contains the `url_to_page_type` CTE.                                                                                                          |
| `seo_data.py` (SEO Clarity)            | `app/`       | Date + domain + device params                       | Date × Keyword × Domain          | Keyword ranking data from SEO Clarity. Currently used in `pages/organic_rankings.py`.                                                                                                     |


### Campaign Bucket Mapping (Paid Search)

The Paid Query uses campaign naming conventions to classify campaigns into buckets. This mapping is critical for paid search analysis:


| Bucket          | Campaign Name Pattern                                                     | Description                        |
| --------------- | ------------------------------------------------------------------------- | ---------------------------------- |
| Supplier        | `d:Supplier`, `Partner`, `d:Competitor`, `d:CustomerService`, `d:Utility` | Branded supplier terms             |
| Aggregator      | `d:Aggregator`                                                            | Power to Choose / aggregator terms |
| Brand           | `Brand`                                                                   | Our own brand terms                |
| Companies       | `d:Companies`                                                             | Energy/electric company terms      |
| Generic         | `d:Generic`, `d:LowPerformance`, `d:TrueBroad`                            | Generic electricity terms          |
| Geo             | `d:Geo`                                                                   | City/location-specific terms       |
| Price Sensitive | `d:PriceSens`                                                             | "Cheap" keywords                   |
| Spanish         | `d:Spanish`                                                               | Spanish-language campaigns         |
| Rates           | `4000148`                                                                 | Rate-specific campaigns            |
| NoDeposit       | `4000150`                                                                 | No-deposit campaigns               |
| PMax            | `pMax`                                                                    | Performance Max campaigns          |
| Other           | (fallthrough)                                                             | Unclassified                       |


Site is also encoded: `a:CTXP`, `a:SOE`, `a:TXER`, `a:CHOO`.

### SEO Clarity — Current Understanding and Gaps

`seo_fact_clarity_keywords_rankings_json` tracks keyword-level ranking positions over time, weighted by `avg_search_volume`. The existing `seo_data.py` module can:

- Fetch rankings by domain, device, date range
- Compute search-volume-weighted average rank
- Build time series of weighted rank by domain
- Build position distribution (buckets: 1, 2, 3-5, 6-10, 11-20, 20+)
- Build page-level scorecards comparing current vs prior period

**What's unclear / needs investigation**:

- How frequently is this table refreshed? (Daily? Weekly?)
- What is the keyword universe? Is it the same across all domains?
- `avg_search_volume = 99` is treated as 0 (likely a sentinel) — is this documented?
- The `keyword_tags` field could be useful for bucketing but its values are unknown
- Relationship between SEO Clarity rankings and GSC rankings (they measure different things — Clarity is tracked keywords, GSC is actual query impressions)

**Recommendation**: Before building the SEO Clarity section into the app, run an exploratory analysis to catalog the keyword universe, tag values, and refresh cadence. Until then, the GSC-based analysis (impressions, clicks, CTR, impression-weighted rank) is the more reliable signal for top-of-funnel visibility.

### Data Source Selection Rules (for the AI)


| Question Type                       | Primary Source                           | Secondary Source                                | Never Use                     |
| ----------------------------------- | ---------------------------------------- | ----------------------------------------------- | ----------------------------- |
| "How are we pacing?"                | `rpt_texas_daily_pacing`                 | —                                               | Session-level                 |
| "What's our revenue?"               | `finance_query`                          | `rpt_texas_daily_pacing`                        | Session-level                 |
| "Why did VC change?"                | `session_level_query` (decomposition)    | —                                               | Finance (too aggregated)      |
| "How are initiatives performing?"   | `session_level_query` (initiative flags) | —                                               | Finance (no initiative flags) |
| "What happened to SEO traffic?"     | GSC d_5 (total organic clicks/impr)      | `session_level_query` (filtered to Organic)     | d_1 (totals won't match GSC)  |
| "Which page types lost clicks?"     | GSC d_3 (page-level drill-down)          | SEO Clarity (ranking supplement)                | d_5 (no page column)          |
| "Which campaigns are driving paid?" | `Paid Query.sql` / `paidsearch_campaign` | `session_level_query` (filtered to Paid Search) | Finance (no campaign detail)  |
| "What's our ranking trend?"         | SEO Clarity (when validated)             | GSC impression-weighted rank                    | —                             |


---

## Revenue Waterfall Specification

### Formula

```
Total Revenue = Phone Revenue + Cart Revenue

Phone Revenue = Sessions × Site RR × Site Conversion × Phone GCV/Order
Cart Revenue  = Sessions × Cart RR × Cart Conversion × Cart GCV/Order
```

### Sequential Substitution Method

For each factor, substitute the actual value (holding all prior factors at actual, all subsequent factors at plan):

```
Plan Revenue = plan_phone + plan_cart

Step 1: Swap Sessions → Actual
  impact = (A.sessions × P.site_rr × P.site_conv × P.phone_gcv
          + A.sessions × P.cart_rr × P.cart_conv × P.cart_gcv) - Plan Revenue

Step 2: Swap Site RR → Actual (phone path only)
  impact = A.sessions × A.site_rr × P.site_conv × P.phone_gcv
         - A.sessions × P.site_rr × P.site_conv × P.phone_gcv

Step 3: Swap Site Conversion → Actual (phone path only)
  impact = A.sessions × A.site_rr × A.site_conv × P.phone_gcv
         - A.sessions × A.site_rr × P.site_conv × P.phone_gcv

Step 4: Swap Phone GCV/Order → Actual
  impact = A.sessions × A.site_rr × A.site_conv × A.phone_gcv
         - A.sessions × A.site_rr × A.site_conv × P.phone_gcv

Steps 5-7: Same pattern for Cart RR, Cart Conversion, Cart GCV/Order

Validation: Sum of all 7 impacts = Pacing Revenue - Plan Revenue
```

### Visualization

Plotly waterfall chart:

- Starting bar: Plan Revenue (green)
- 7 factor bars: positive (green, increasing) or negative (red, decreasing)
- Ending bar: Pacing Revenue (blue)
- Labels on each bar: dollar value and % of total gap

---

## Implementation Plan

### Phase 0: Data Architecture Foundation (Week 1)

1. Create `app/data_registry.py` — central module that documents every data source, its trust level, and which questions it answers. The AI system prompt and the Cursor rules both reference this.
2. Validate session count alignment between `finance_query` and `session_level_query` for the last 3 months. Document any delta and root cause.
3. Run SEO Clarity exploratory analysis (refresh cadence, keyword universe, tag values). Document findings.
4. Parameterize the Paid Query — replace `:grain` with Python f-string, add date range parameters.

### Phase 1: Restructure into Tabs

1. Refactor `streamlit_app.py` into tabbed layout
2. Move existing Sections 1-7 into Tab 1 (Overview)
3. Add the Revenue Waterfall as Section 4 in Tab 1
4. Move "Ask the Analyst" to its own Tab 4
5. Stub out Tab 2 (Organic) and Tab 3 (Paid Search)

### Phase 2: Revenue Waterfall Implementation 

1. Create `app/waterfall.py` with the sequential substitution computation
2. Build Plotly waterfall chart rendering
3. Wire into both Overview tab (all channels) and channel deep-dive tabs
4. Validate: waterfall impacts must sum exactly to the revenue gap

### Phase 3: Organic Deep Dive Tab

1. Port GSC queries from `seo_landing_page_type_analysis.ipynb` into `app/seo_data.py` (extend existing)
2. Build the 4-panel GSC visibility chart
3. Build click decomposition section
4. Build landing page type breakdown (reuse `url_to_page_type` CTE)
5. Build session funnel by page type

### Phase 4: Paid Search Deep Dive Tab

1. Create `app/paid_search_data.py` — parameterized version of `docs/Paid Query.sql` with campaign bucket mapping
2. Build campaign bucket performance table (Section 3)
3. Build campaign bucket revenue waterfall (Section 4)
4. Build cost/efficiency trended charts (Section 5)
5. Build campaign bucket drill-down view (Section 6)
6. Build Paid Search PoP decomposition with campaign bucket as a dimension (Section 7)
7. Stub auction performance section for future Google Ads integration (Section 8)

### Phase 5: Enhanced Business Knowledge 

1. Expand system prompt in `narrative.py` with full business context (funnel, initiatives, seasonality, channel characteristics)
2. Update `analyst_tools.py` system prompt with the Data Source Selection Rules from the PRD
3. Add reconciliation awareness: when the AI presents session-level metrics alongside finance metrics, it should note the source and caveats
4. Ensure missing business context docs are created (`operational_overview.docx`, `personalization_initiatives.docx`)

### Phase 6: Polish & Testing 

1. Good useful app 
2. Error handling for partial data (e.g., GSC not available for Paid Search tab)
3. Performance optimization (caching, lazy loading of deep-dive tabs)
4. Data reconciliation checks built into the footer (session count comparison between finance and session-level)
5. User testing with 2-3 analysts

---

## File Structure (Target)

```
app/
  streamlit_app.py          # Main entry: sidebar + tab routing
  config.py                 # KPI registry, dimensions, constants
  data_registry.py          # NEW: Central data source documentation + trust levels
  data.py                   # Session-level data fetch
  finance_data.py           # Finance actuals + plan/pacing
  kpi_engine.py             # KPI computation
  decomposition.py          # Mix/rate decomposition
  waterfall.py              # NEW: Revenue waterfall computation (sequential substitution)
  narrative.py              # LLM narrative generation (expanded business context)
  analyst_tools.py          # Chat tools (pandas + SQL, with reconciliation awareness)
  seo_data.py               # SEO Clarity keyword rankings + GSC visibility analysis
  paid_search_data.py       # NEW: Paid Search campaign-level data + bucket mapping
  time_periods.py           # Time period resolution
  pages/
    overview.py             # NEW: Tab 1 rendering
    organic_deep_dive.py    # NEW: Tab 2 rendering (replaces organic_rankings.py)
    paid_search_deep_dive.py# NEW: Tab 3 rendering
    analyst_chat.py         # NEW: Tab 4 rendering
docs/
  Paid Query.sql            # Existing: campaign-level paid search query
  business_context/
    full_funnel.docx        # Existing: KPI definitions and funnel logic
    operational_overview.docx
    personalization_initiatives.docx
queries/                    # NEW: Parameterized query files (migrated from root)
  finance_query.sql
  session_level_query.sql
  paid_query.sql
  gsc_summary.sql
  gsc_page_type.sql
.cursor/
  rules/
    seo-reporting-agent.mdc # Agent protocol for SEO questions
```

---

## Success Criteria

1. Any analyst can select KPI + Channel + Date and see pacing, PoP, waterfall in <10 seconds
2. Revenue waterfall impacts sum to exact revenue gap (validated to $1)
3. Organic deep dive shows GSC visibility trends, click decomposition, and page-type breakdown without manual notebook execution
4. Paid Search deep dive shows campaign bucket performance, revenue waterfall by bucket, and drill-down to individual bucket trends
5. LLM narratives reference specific drivers and explain causality ("Cart Conversion improved because SSN Submit Rate increased in Direct traffic") rather than restating numbers
6. "Ask the Analyst" can answer ad-hoc questions like "which page type lost the most clicks in April?" or "which campaign bucket drove the CPC increase?" by querying data directly
7. When session-level metrics differ from finance metrics, the tool clearly labels which source is being used and explains the discrepancy if asked

---

## Out of Scope (v2)

- Slack bot integration (future — extract agent protocol into Python module first)
- Google Ads API integration (stub only; depends on API access)
- Automated scheduling / daily email reports
- User authentication / role-based access
- Multi-market support (Texas only for now)
- SEO Clarity deep integration (needs exploratory analysis first — use GSC as primary SEO data source)

