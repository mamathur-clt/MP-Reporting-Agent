# Energy Marketplace — Funnel KPI Driver Analysis Tool

A Streamlit app that explains period-over-period changes in energy marketplace funnel KPIs, designed for recurring weekly business reviews (Tuesday WoW, Friday WTD).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fill in credentials in .env
#    DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH, OPENAI_API_KEY

# 3. Run the app
streamlit run app/streamlit_app.py
```

## What This Tool Does

Given a KPI and a time comparison mode, the tool:

1. **Runs the full session-level funnel query** against Databricks (the exact query in `session_level_query`)
2. **Computes the KPI** for both the current and prior period
3. **Decomposes the change** into per-segment contributions across multiple dimensions
4. **Shows a horizontal bar chart** of the top positive and negative drivers with clean labels
5. **Evaluates initiative performance** (LP JO, Grid JO, FMP) via model-vs-holdout comparison
6. **Generates an AI-written summary** via GPT-4o, suitable for copy-pasting into a weekly review

## Supported KPIs

All formulas are derived from the Full Funnel business specification (`docs/business_context/full_funnel.docx`).

| KPI | Formula | Description |
|-----|---------|-------------|
| **ZLUR** | `SUM(zip_entry) / SUM(session)` | ZIP Lookup Rate — product-view proxy for ZIP intent |
| **Cart RR** | `SUM(has_cart) / SUM(session)` | Cart Reach Rate — share of sessions entering cart |
| **SSN Submit Rate** | `SUM(cart_ssn_done) / SUM(has_cart)` | Among cart sessions, share completing SSN/credit step |
| **Conversion After Credit** | `SUM(cart_order) / SUM(cart_ssn_done)` | Post-credit cart order rate |
| **Cart Conversion** | `SUM(cart_order) / SUM(has_cart)` | Overall cart conversion (= SSN Submit × Post-Credit) |
| **Cart VC** | `SUM(cart_order) / SUM(session)` | Cart visit conversion — end-to-end digital |
| **Phone RR** | `SUM(queue_call) / SUM(session)` | Phone reach rate (queue calls) |
| **Phone VC** | `SUM(phone_order) / SUM(session)` | Phone visit conversion |
| **VC** | `SUM(cart_order + phone_order) / SUM(session)` | Total visit conversion (cart + phone) — the most important KPI |

### Key identity

**Cart Conversion = SSN Submit Rate × Conversion After Credit**

This decomposition separates pre-credit friction from post-credit eligibility/routing performance. The app surfaces this automatically when analyzing Cart Conversion or VC.

## Decomposition Approach

### Mix-Shift / Rate-Change Framework

For any KPI = Σ(w_s × r_s) across segments of a dimension, the period-over-period change decomposes as:

```
contribution_s = (Δw_s × r̄_s) + (Δr_s × w̄_s)
                  ───────────     ───────────
                  mix effect      rate effect
```

Where:
- **w_s** = share of the KPI's denominator falling in segment s
- **r_s** = KPI rate within segment s
- **Δ** = current minus prior
- **r̄, w̄** = average of current and prior values

This is the Shapley-value-consistent additive decomposition: contributions across all segments sum exactly to the total ΔR. It answers questions like *"Paid Search contributed +15 bps because its share of sessions grew and its conversion rate improved."*

### Dimensions analyzed

Derived from the session-level query output and business context:

| Dimension | Why it matters |
|-----------|---------------|
| `website` | 4 marketplace sites (CTXP, SOE, Choose TX, Choose NTX, TXER) with different traffic profiles |
| `marketing_channel` | Paid Search, Organic, Direct, AI Search, Affiliate, etc. — major volume and rate driver |
| `mover_switcher` | Movers vs switchers have fundamentally different conversion patterns |
| `device_type` | Mobile vs desktop affects funnel progression |
| `landing_page_type` | Provider, Grid, Resources pages — reflects user intent |
| `first_partner_name` | Which provider the user first considered — affects credit pass rates |
| `_initiative_label` | LP JO / Grid JO / FMP model vs holdout — personalization impact |

### Initiative Analysis

Per the Personalization Initiatives business context:

- **LP JO / Grid JO**: Model vs holdout comparison with journey consistency enforced. Sessions that flip between model and holdout across LP and grid are excluded (filtered in the query via `initiative_master`).
- **FMP**: Compared against LP holdout as its control population (FMP is entered from the landing page).
- **Counterfactual delta**: For each model group, "how much better/worse is the model rate vs the holdout rate?"

## Assumptions and Design Decisions

1. **`session` column = 1 always**: Per the business spec, every row is one session. SUM(session) = row count.

2. **`total_order` = `cart_order` + `phone_order`**: VC uses this derived field. A session can have both a cart and phone order (rare but possible).

3. **Numeric coercion**: All flag columns are coerced to numeric with NaN → 0. This handles null values from left joins in the upstream query (e.g., a session with no cart data has cart_order = NULL → 0).

4. **Date filtering uses `session_start_date_est`**: The primary date dimension for period assignment.

5. **Initiative labels are mutually exclusive per session**: A session gets one label based on priority: FMP > LP+Grid JO > LP JO only > Grid JO only > No Initiative. This follows the query's `initiative_master` CTE logic.

6. **Decomposition is additive but not causal**: Mix-shift analysis explains *what changed compositionally* but does not prove causation. A rate change in Paid Search might itself be caused by a provider change or a model update.

7. **Query dates are parameterized**: The tool injects the required date range into the `cte_variables` CTE of the session-level query, so only the needed data is pulled.

8. **Results cached for 1 hour**: Streamlit's `@st.cache_data` prevents re-querying Databricks on every UI interaction.

## Time Comparison Modes

| Mode | Current period | Prior period | Use case |
|------|---------------|-------------|----------|
| **WoW** | Last full Mon–Sun | The Mon–Sun before that | Standard weekly review |
| **WTD (Tue review)** | Yesterday (Monday) | Prior Monday | Tuesday morning check-in |
| **WTD (Fri review)** | This Mon–Thu | Prior Mon–Thu | Friday WTD summary |
| **MoM** | This month to yesterday | Full prior month | Monthly review |
| **Custom** | User-defined | User-defined | Ad hoc analysis |

## Project Structure

```
MP-Reporting-Agent/
├── app/
│   ├── __init__.py
│   ├── config.py            # KPI definitions, dimension lists, time modes
│   ├── data.py              # Databricks connection, query execution, caching
│   ├── time_periods.py      # Time period resolution logic
│   ├── kpi_engine.py        # KPI computation from DataFrames
│   ├── decomposition.py     # Mix-shift / rate-change + initiative analysis
│   ├── narrative.py         # Written summary generation
│   └── streamlit_app.py     # Main Streamlit UI
├── docs/
│   └── business_context/
│       ├── full_funnel.docx
│       ├── operational_overview.docx
│       └── personalization_initiatives.docx
├── notebooks/
│   ├── databricks_auth_and_query.ipynb
│   └── query_sessions.ipynb
├── session_level_query       # The full 730-line SQL query
├── .env                      # Databricks credentials (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Limitations and Next Steps

### Current limitations

- **Single-dimension decomposition**: Each dimension is decomposed independently. Cross-dimensional interactions (e.g., "Paid Search on CTXP specifically") are not captured. A future version could support two-way decompositions.
- **No statistical significance testing**: Initiative model-vs-holdout comparisons show rate differences but don't include confidence intervals or p-values. For formal experiment evaluation, a proper statistical framework should be added.
- **LLM narrative cost**: Each summary generation makes one GPT-4o API call (~800 tokens output). Falls back to a structured template if the API call fails.
- **No GCV / revenue decomposition**: The tool focuses on conversion rates. GCV (gross contract value) decomposition would add a revenue lens.
- **Query runtime**: The full session-level query can take 30–90 seconds on the SQL warehouse. The 1-hour cache mitigates this for interactive use.

### Planned next steps

1. **Two-way decomposition**: e.g., marketing_channel × website interaction effects
2. **Daily trend charts**: Show the KPI trend line over the current and prior period
3. **Credit quality deep-dive**: Decompose Conversion After Credit by credit_fail / volt_fail / qual_fail rates
4. **Anomaly detection**: Flag dimensions where the rate change exceeds historical norms
5. **LLM model selection**: Allow choosing between GPT-4o and GPT-4o-mini for cost control
6. **Scheduled reports**: Auto-run and email/Slack results on Tuesday and Friday mornings
