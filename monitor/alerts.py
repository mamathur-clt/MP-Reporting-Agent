"""
Partner Health Monitor — alerting.

Formats Slack Block Kit messages, generates Plotly trend charts,
posts alerts via webhook, and manages the alert log.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

from monitor.anomaly import AnomalyRecord, record_cooldown
from monitor.config import (
    ALERT_LOG_PATH,
    CHART_DIR,
    METRICS,
    SLACK_WEBHOOK_URL,
)


# ── Chart generation ──────────────────────────────────────────────────────

def generate_trend_chart(
    partner: str,
    metric_key: str,
    history_metrics: pd.DataFrame,
    anomaly: AnomalyRecord,
    days: int = 7,
) -> str | None:
    """
    Create a 7-day trend chart for the partner × metric with the baseline
    band shaded and the anomaly point highlighted.  Saves to CHART_DIR
    and returns the file path, or None on failure.
    """
    mdef = METRICS.get(metric_key)
    if mdef is None or history_metrics.empty:
        return None

    subset = history_metrics[
        (history_metrics["partner"] == partner)
        & (history_metrics["metric_key"] == metric_key)
    ].copy()

    if "_date" not in subset.columns or subset.empty:
        return None

    subset["_date"] = pd.to_datetime(subset["_date"])
    subset = subset.sort_values("_date")

    # Keep last N days
    cutoff = subset["_date"].max() - pd.Timedelta(days=days)
    subset = subset[subset["_date"] >= cutoff]

    if subset.empty:
        return None

    daily = subset.groupby("_date").agg(
        rate=("rate", "mean"),
        volume=("denominator_sum", "sum"),
    ).reset_index()

    fig = go.Figure()

    # Baseline band
    b_mean = anomaly.baseline_mean
    b_std = anomaly.baseline_std
    fig.add_hrect(
        y0=max(0, b_mean - 2 * b_std),
        y1=min(1, b_mean + 2 * b_std),
        fillcolor="rgba(100,149,237,0.15)",
        line_width=0,
        annotation_text="Baseline ±2σ",
        annotation_position="top left",
    )

    fig.add_hline(
        y=b_mean,
        line_dash="dash",
        line_color="cornflowerblue",
        annotation_text=f"Baseline {b_mean:.1%}",
    )

    # Trend line
    fig.add_trace(
        go.Scatter(
            x=daily["_date"],
            y=daily["rate"],
            mode="lines+markers",
            name=mdef.name,
            line=dict(color="#2c3e50", width=2),
            marker=dict(size=6),
        )
    )

    # Anomaly point
    fig.add_trace(
        go.Scatter(
            x=[daily["_date"].iloc[-1]],
            y=[anomaly.current_rate],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=14, symbol="x"),
        )
    )

    fig.update_layout(
        title=f"{partner} — {mdef.name}",
        yaxis_title=mdef.name,
        yaxis_tickformat=".0%",
        xaxis_title="",
        template="plotly_white",
        height=320,
        width=600,
        margin=dict(l=50, r=20, t=50, b=30),
        showlegend=False,
    )

    os.makedirs(CHART_DIR, exist_ok=True)
    safe_name = f"{partner}_{metric_key}_{datetime.now():%Y%m%d_%H%M%S}.png"
    path = os.path.join(CHART_DIR, safe_name)

    try:
        fig.write_image(path, engine="kaleido")
    except Exception:
        path = path.replace(".png", ".html")
        fig.write_html(path)

    return path


# ── Slack message formatting ─────────────────────────────────────────────

def format_slack_message(
    anomaly: AnomalyRecord,
    credit_dist: pd.DataFrame | None = None,
    qual_errors: pd.DataFrame | None = None,
    child_breakdown: pd.DataFrame | None = None,
) -> dict:
    """Build a Slack Block Kit payload for the anomaly."""
    mdef = METRICS.get(anomaly.metric_key)
    metric_name = mdef.name if mdef else anomaly.metric_key

    direction = "dropped" if anomaly.pp_drop < 0 else "spiked"
    scope_label = f"[{anomaly.partner} — Parent Group]" if anomaly.level == "parent" else anomaly.partner

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"⚠️  {scope_label} — {metric_name} Alert",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{metric_name}* {direction} to *{anomaly.current_rate:.1%}* "
                    f"(baseline: {anomaly.baseline_mean:.1%}, z-score: {anomaly.z_score:+.1f})\n"
                    f"Volume: {anomaly.volume} cart sessions  |  "
                    f"Drop: {anomaly.pp_drop:+.1%} pp"
                ),
            },
        },
    ]

    # Child-partner breakdown for parent-level alerts
    if anomaly.level == "parent" and child_breakdown is not None and not child_breakdown.empty:
        rows = child_breakdown.sort_values("denominator_sum", ascending=False)
        table_lines = ["*Partner Breakdown:*", "```", "Provider             |  Rate  | Volume"]
        table_lines.append("-" * 44)
        for _, r in rows.iterrows():
            rate_str = f"{r['rate']:.1%}" if pd.notna(r.get("rate")) else "N/A"
            table_lines.append(
                f"{str(r['partner'])[:21]:21s}| {rate_str:>6s} | {int(r.get('denominator_sum', 0)):>5d}"
            )
        table_lines.append("```")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(table_lines)},
        })

    # Credit score distribution context
    if credit_dist is not None and not credit_dist.empty:
        partner_dist = credit_dist[credit_dist["partner"] == anomaly.partner]
        if not partner_dist.empty:
            rows = partner_dist.sort_values("count", ascending=False).head(6)
            table_lines = ["```", "Score Bucket     | Pass | Fail | Total"]
            table_lines.append("-" * 44)
            for _, r in rows.iterrows():
                table_lines.append(
                    f"{str(r['CreditScoreBucketCR']):17s}| {int(r['credit_pass_count']):4d} "
                    f"| {int(r['credit_fail_count']):4d} | {int(r['count']):5d}"
                )
            table_lines.append("```")
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(table_lines)},
            })

    # Qual error breakdown context
    if qual_errors is not None and not qual_errors.empty:
        partner_errs = qual_errors[qual_errors["partner"] == anomaly.partner]
        if not partner_errs.empty:
            rows = partner_errs.sort_values("count", ascending=False).head(5)
            table_lines = ["```", "Error Reason           | Count | Share"]
            table_lines.append("-" * 44)
            for _, r in rows.iterrows():
                table_lines.append(
                    f"{str(r['first_qual_error_reason'])[:23]:23s}| {int(r['count']):5d} "
                    f"| {r['share']:.0%}"
                )
            table_lines.append("```")
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(table_lines)},
            })

    # Conversion-specific context: flag if credit pass is stable
    if anomaly.metric_key == "conversion_after_credit":
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    ":mag: *Diagnostic*: Credit pass rate may be stable for this provider. "
                    "If so, the drop in post-credit conversion suggests a non-credit issue — "
                    "e.g. provider API error, enrollment flow breakage, or plan availability change."
                ),
            },
        })

    blocks.append({"type": "divider"})

    return {"blocks": blocks}


# ── Slack delivery ────────────────────────────────────────────────────────

def send_slack_alert(
    message: dict,
    chart_path: str | None = None,
) -> bool:
    """
    POST alert to Slack webhook. Returns True on success.
    If no webhook is configured, logs to console and returns False.
    """
    if not SLACK_WEBHOOK_URL:
        print("[monitor] Slack webhook not configured — alert logged locally only.")
        _log_alert(message)
        return False

    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            json=message,
            timeout=15,
        )
        resp.raise_for_status()
        _log_alert(message)
        return True
    except Exception as exc:
        print(f"[monitor] Slack delivery failed: {exc}")
        _log_alert(message)
        return False


# ── Local alert log ───────────────────────────────────────────────────────

def _log_alert(message: dict) -> None:
    """Append alert to a local JSONL log file."""
    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
    }
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Public helper combining chart + message + send ────────────────────────

def fire_alert(
    anomaly: AnomalyRecord,
    history_metrics: pd.DataFrame,
    credit_dist: pd.DataFrame | None = None,
    qual_errors: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
) -> None:
    """End-to-end: generate chart, format message, send, record cooldown."""
    chart_path = generate_trend_chart(
        anomaly.partner,
        anomaly.metric_key,
        history_metrics,
        anomaly,
    )

    # For parent-level alerts, compute per-child breakdown
    child_breakdown = None
    if anomaly.level == "parent" and current_df is not None:
        from monitor.metrics import get_child_partner_breakdown
        child_breakdown = get_child_partner_breakdown(
            current_df, anomaly.partner, anomaly.metric_key,
        )

    msg = format_slack_message(
        anomaly,
        credit_dist=credit_dist,
        qual_errors=qual_errors,
        child_breakdown=child_breakdown,
    )

    send_slack_alert(msg, chart_path=chart_path)
    record_cooldown(anomaly.partner, anomaly.metric_key)
