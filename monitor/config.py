"""
Partner Health Monitor — configuration.

Central place for thresholds, metric definitions, environment wiring,
and Slack webhook settings.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Databricks credentials (shared with the main app) ─────────────────────
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")

# ── Slack ──────────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ── Anomaly detection thresholds ──────────────────────────────────────────
Z_SCORE_THRESHOLD = -2.5          # standard deviations below baseline
PP_FLOOR = 0.05                   # minimum absolute drop (5 pp) to alert
MIN_VOLUME = 20                   # minimum cart sessions in window
BASELINE_DAYS = 14                # rolling window for baseline stats
COOLDOWN_HOURS = 6                # suppress repeat alerts for same partner×metric

# ── Scheduling ────────────────────────────────────────────────────────────
CHECK_INTERVAL_HOURS = 1          # how often the daemon loop runs
DATA_START_DATE = "2026-01-01"    # earliest date for backfill / baseline

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CART_QUERY_PATH = os.path.join(REPO_ROOT, "queries", "cart_session_level_query.txt")
ALERT_LOG_PATH = os.path.join(REPO_ROOT, "monitor", "alert_log.jsonl")
COOLDOWN_LOG_PATH = os.path.join(REPO_ROOT, "monitor", "cooldown_log.json")
CHART_DIR = os.path.join(REPO_ROOT, "monitor", "charts")


# ── Metric definitions ────────────────────────────────────────────────────

@dataclass
class MetricDef:
    key: str
    name: str
    numerator: str
    denominator: str
    partner_col: str
    description: str = ""
    higher_is_better: bool = True


# ── Parent company mapping ────────────────────────────────────────────────
# Partners that share infrastructure — aggregate for more robust detection.

PARENT_COMPANIES: dict[str, list[str]] = {
    "Vistra":   ["TXU Energy", "TriEagle Energy", "4Change Energy", "Express Energy", "Veteran Energy"],
    "NRG":      ["Reliant", "Cirro Energy", "Green Mountain", "Discount Power", "Direct Energy"],
    "NextEra":  ["Frontier Utilities", "Gexa Energy"],
}

# Reverse lookup: partner name → parent company
PARTNER_TO_PARENT: dict[str, str] = {}
for _parent, _children in PARENT_COMPANIES.items():
    for _child in _children:
        PARTNER_TO_PARENT[_child] = _parent


METRICS: dict[str, MetricDef] = {
    "credit_pass": MetricDef(
        key="credit_pass",
        name="Credit Pass Rate",
        numerator="credit_pass_flag",
        denominator="has_credit_run",
        partner_col="first_run_provider_name",
        description="Share of credit-run cart sessions that passed the provider threshold.",
    ),
    "qual_pass": MetricDef(
        key="qual_pass",
        name="Qual Pass Rate",
        numerator="qual_pass_flag",
        denominator="has_qual_result",
        partner_col="midflow_provider",
        description="Share of qual-checked cart sessions that passed provider qualification.",
    ),
    "volt_pass": MetricDef(
        key="volt_pass",
        name="Volt Pass Rate",
        numerator="volt_pass_flag",
        denominator="has_cart_session",
        partner_col="first_run_provider_name",
        description="Share of cart sessions that passed Volt (fraud/identity) checks.",
    ),
    "conversion_after_credit": MetricDef(
        key="conversion_after_credit",
        name="Conversion After Credit",
        numerator="cart_order",
        denominator="ssn_completion",
        partner_col="first_partner_name",
        description="Among SSN-submit sessions, share that completed a cart order.",
    ),
}
