# MP Reporting Agent

An internal reporting toolkit for the Energy Marketplace team. Includes a
Streamlit dashboard, a Slack bot, and an automated anomaly monitor — all
backed by Databricks SQL.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials (copy .env.example or create .env)
#    Required: DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH
#    Optional: OPENAI_API_KEY, SLACK_BOT_TOKEN, SLACK_APP_TOKEN

# 3. Run the Streamlit app
streamlit run app/streamlit_app.py
```

## Streamlit Dashboard

The main interface is a multi-tab Streamlit app for weekly and monthly
business reviews.

### Tabs

| Tab | What it does |
|-----|-------------|
| **Overview** | KPI scorecards, mix-shift decomposition, waterfall charts, and an AI-generated narrative for any funnel metric (VC, Cart Conversion, ZLUR, etc.) across configurable time windows |
| **Deep Dive — Organic** | GSC trends, keyword rankings, page-type traffic, page-1 churn analysis, and an organic session funnel |
| **Deep Dive — Paid Search** | Campaign-bucket rollups, spend/session/order breakdowns, and period-over-period comparisons |
| **Ask the Analyst** | Natural-language Q&A powered by OpenAI with direct Databricks SQL access for ad-hoc questions |

### Time Modes

| Mode | Current period | Prior period |
|------|---------------|-------------|
| WoW | Last full Mon–Sun | The Mon–Sun before that |
| WTD (Tue review) | Yesterday (Monday) | Prior Monday |
| WTD (Fri review) | Mon–Thu this week | Mon–Thu last week |
| MoM | Month-to-date through yesterday | Full prior month |
| Custom | User-defined | User-defined |

### Data Freshness

All queries use `@st.cache_data` with TTLs of 30–60 minutes. After
the TTL expires, the next user interaction triggers a fresh query against the
SQL warehouse. Changing the date range or filters in the sidebar always
fetches new data.

## Slack Bot

A Slack bot (`bot/`) that responds to mentions and DMs with SEO and funnel
analysis via an OpenAI tool-calling agent. Supports direct SQL queries and
chart generation.

```bash
python -m bot
```

Requires `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, and `OPENAI_API_KEY`.

## Partner Health Monitor

An anomaly detection service (`monitor/`) that compares recent hourly cart
metrics against a historical baseline and fires Slack alerts when thresholds
are breached.

```bash
# One-shot check
python -m monitor

# Recurring daemon (hourly)
python -m monitor.run --daemon

# Backfill historical anomalies
python -m monitor.run --backfill 2026-01-01
```

## Project Structure

```
MP-Reporting-Agent/
├── app/
│   ├── streamlit_app.py        # Main entry point
│   ├── db.py                   # Centralized Databricks connection (PAT + OAuth)
│   ├── tabs/                   # Tab modules (overview, organic, paid, chat)
│   ├── config.py               # KPI definitions, dimensions, time modes
│   ├── data.py                 # Session funnel data (Databricks)
│   ├── seo_data.py             # GSC / organic ranking data
│   ├── paid_search_data.py     # Paid search campaign data
│   ├── finance_data.py         # Finance actuals and plan pacing
│   ├── time_periods.py         # Dynamic period resolution
│   ├── kpi_engine.py           # KPI computation
│   ├── decomposition.py        # Mix-shift / rate-change analysis
│   ├── narrative.py            # AI narrative generation
│   ├── waterfall.py            # Waterfall chart builder
│   ├── seo_diagnostic.py       # SEO diagnostic utilities
│   ├── analyst_tools.py        # Tool definitions for the analyst chat
│   └── app_context.py          # Shared state across tabs
├── bot/
│   ├── app.py                  # Slack Bolt entry point
│   ├── agent.py                # OpenAI tool-calling agent
│   ├── tools/                  # SEO and SQL tool implementations
│   └── prompts/                # System prompts
├── monitor/
│   ├── run.py                  # Scheduler / CLI entry point
│   ├── data.py                 # Cart query execution
│   ├── anomaly.py              # Z-score anomaly detection
│   ├── alerts.py               # Slack webhook alerts
│   └── config.py               # Thresholds and paths
├── queries/
│   ├── session_level_query     # Full session funnel SQL
│   ├── finance_query           # Finance actuals SQL
│   ├── plan_query              # Plan/pacing SQL
│   └── cart_session_level_query.txt  # Cart-level query (monitor)
├── docs/
│   ├── Paid Query.sql          # Paid search SQL template
│   ├── prd_v2.md               # Product requirements
│   └── business_context/       # Business spec documents
├── notebooks/                  # Exploratory analysis notebooks
├── assets/                     # Static images (SEO flowcharts)
├── app.yaml                    # Databricks Apps launch config
├── requirements.txt
└── .env                        # Credentials (git-ignored)
```

## Deploying to Databricks Apps

The app is ready to deploy as a Databricks App. The `app.yaml` at the repo
root provides the launch command.

### Steps

1. In the Databricks workspace go to **Compute > Apps > Create app**.
2. Choose **Custom**, name the app, and create it.
3. Once compute starts, click **Deploy** and point to the Git repo (branch `main`)
   or upload the workspace folder.
4. Configure environment variables for the app (see table below).
5. Grant the app's **service principal** these permissions:
   - `SELECT` on all Unity Catalog tables the queries reference
     (`energy_prod.energy.*`, `lakehouse_production.common.*`,
     `lakehouse_production.energy.*`, `energy_prod.data_science.*`,
     `energy_prod.fivetran_finance.*`)
   - `CAN USE` on the SQL warehouse in `DATABRICKS_HTTP_PATH`

### Auth modes

The app auto-detects which auth to use (`app/db.py`):

- **Databricks Apps** — when `DATABRICKS_CLIENT_ID` and
  `DATABRICKS_CLIENT_SECRET` are injected by the platform, OAuth via the
  Databricks SDK is used. No PAT needed.
- **Local development** — falls back to `DATABRICKS_TOKEN` from `.env`.

### Updating after deployment

1. Make changes locally and test with `streamlit run app/streamlit_app.py`.
2. `git push` to the repo.
3. In the Databricks Apps UI click **Deploy** (or run
   `databricks apps deploy <app-name>` from the CLI).

## Environment Variables

| Variable | Required by | Purpose |
|----------|------------|---------|
| `DATABRICKS_HOST` | All (local dev) | Workspace URL |
| `DATABRICKS_TOKEN` | All (local dev) | Personal access token |
| `DATABRICKS_HTTP_PATH` | All | SQL warehouse path (needed in both local and Databricks Apps) |
| `DATABRICKS_CLIENT_ID` | Databricks Apps | Auto-injected by platform |
| `DATABRICKS_CLIENT_SECRET` | Databricks Apps | Auto-injected by platform |
| `OPENAI_API_KEY` | App (chat tab), Bot | GPT-4o for narratives and Q&A |
| `SLACK_BOT_TOKEN` | Bot | Slack bot OAuth token |
| `SLACK_APP_TOKEN` | Bot | Slack app-level token (Socket Mode) |
| `SLACK_WEBHOOK_URL` | Monitor | Webhook for anomaly alerts |
