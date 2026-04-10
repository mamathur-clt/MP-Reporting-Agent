"""
Partner Health Monitor — anomaly detection.

Computes a rolling baseline (mean + std) for each partner × metric
using 14 days of history, matched by day-of-week and hour-of-day.
Flags anomalies via z-score + absolute pp drop + volume gate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from monitor.config import (
    BASELINE_DAYS,
    COOLDOWN_HOURS,
    COOLDOWN_LOG_PATH,
    MIN_VOLUME,
    PP_FLOOR,
    Z_SCORE_THRESHOLD,
)
from monitor.metrics import compute_partner_metrics


# ── Result container ──────────────────────────────────────────────────────

@dataclass
class AnomalyRecord:
    partner: str
    metric_key: str
    current_rate: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    pp_drop: float
    volume: int
    detected_at: str  # ISO timestamp
    level: str = "partner"  # "partner" or "parent"
    extra: dict = field(default_factory=dict)


# ── Baseline computation ─────────────────────────────────────────────────

def compute_baseline(
    history_df: pd.DataFrame,
    target_dow: str | None = None,
    target_hour: int | None = None,
) -> pd.DataFrame:
    """
    From a historical cart-session DataFrame compute per-partner × metric
    baseline stats (mean, std, n_windows).

    Optionally filter to a specific day_of_week + hour to capture
    same-period seasonality.
    """
    group_cols = ["_date", "hour", "day_of_week"]
    present = [c for c in group_cols if c in history_df.columns]
    if not present:
        present = []

    metrics_agg = compute_partner_metrics(history_df, group_cols=present, level="both")

    if metrics_agg.empty:
        return pd.DataFrame(
            columns=["partner", "metric_key", "baseline_mean", "baseline_std", "n_windows"]
        )

    if target_dow and "day_of_week" in metrics_agg.columns:
        metrics_agg = metrics_agg[metrics_agg["day_of_week"] == target_dow]
    if target_hour is not None and "hour" in metrics_agg.columns:
        metrics_agg = metrics_agg[metrics_agg["hour"] == target_hour]

    if metrics_agg.empty:
        return pd.DataFrame(
            columns=["partner", "metric_key", "baseline_mean", "baseline_std", "n_windows"]
        )

    grp = ["partner", "metric_key"]
    if "level" in metrics_agg.columns:
        grp.append("level")

    baseline = (
        metrics_agg
        .groupby(grp, dropna=False)
        .agg(
            baseline_mean=("rate", "mean"),
            baseline_std=("rate", "std"),
            n_windows=("rate", "count"),
        )
        .reset_index()
    )
    baseline["baseline_std"] = baseline["baseline_std"].fillna(0)
    return baseline


# ── Anomaly detection ────────────────────────────────────────────────────

def detect_anomalies(
    current_df: pd.DataFrame,
    history_df: pd.DataFrame,
    target_dow: str | None = None,
    target_hour: int | None = None,
    skip_cooldown: bool = False,
) -> list[AnomalyRecord]:
    """
    Compare current-period partner metrics against the rolling baseline
    and return a list of AnomalyRecords for anything that exceeds thresholds.
    """
    current_metrics = compute_partner_metrics(current_df, level="both")
    if current_metrics.empty:
        return []

    baseline = compute_baseline(history_df, target_dow=target_dow, target_hour=target_hour)
    if baseline.empty:
        return []

    merged = current_metrics.merge(
        baseline,
        on=["partner", "metric_key", "level"],
        how="left",
    )

    anomalies: list[AnomalyRecord] = []
    now_iso = datetime.now().isoformat()

    for _, row in merged.iterrows():
        vol = int(row.get("denominator_sum", 0))
        if vol < MIN_VOLUME:
            continue

        rate = row.get("rate", np.nan)
        b_mean = row.get("baseline_mean", np.nan)
        b_std = row.get("baseline_std", np.nan)

        if pd.isna(rate) or pd.isna(b_mean):
            continue

        pp_drop = rate - b_mean

        if b_std and b_std > 0:
            z = (rate - b_mean) / b_std
        else:
            z = 0.0 if abs(pp_drop) < PP_FLOOR else (-10.0 if pp_drop < 0 else 10.0)

        if z <= Z_SCORE_THRESHOLD and abs(pp_drop) >= PP_FLOOR:
            lvl = row.get("level", "partner")
            if not skip_cooldown and _is_on_cooldown(row["partner"], row["metric_key"]):
                continue

            anomalies.append(
                AnomalyRecord(
                    partner=row["partner"],
                    metric_key=row["metric_key"],
                    current_rate=float(rate),
                    baseline_mean=float(b_mean),
                    baseline_std=float(b_std) if not pd.isna(b_std) else 0.0,
                    z_score=float(z),
                    pp_drop=float(pp_drop),
                    volume=vol,
                    detected_at=now_iso,
                    level=str(lvl),
                )
            )

    return anomalies


# ── Hourly-grain anomaly detection (for backfill / dashboard) ─────────────

def detect_hourly_anomalies(
    all_data: pd.DataFrame,
    baseline_days: int = BASELINE_DAYS,
) -> pd.DataFrame:
    """
    Run anomaly detection for every date × hour slot in the data.
    For each slot the baseline is the preceding *baseline_days*, filtered
    to matching day-of-week AND hour-of-day.
    Returns a DataFrame of all anomaly records (useful for dashboard).
    """
    if "_date" not in all_data.columns or "hour" not in all_data.columns:
        return pd.DataFrame()

    all_data = all_data.copy()
    all_data["_date_dt"] = pd.to_datetime(all_data["_date"])
    if "day_of_week" not in all_data.columns:
        all_data["day_of_week"] = all_data["_date_dt"].dt.strftime("%A")

    slots = (
        all_data[["_date_dt", "hour", "day_of_week"]]
        .drop_duplicates()
        .sort_values(["_date_dt", "hour"])
    )

    records: list[dict] = []
    total_slots = len(slots)

    for i, (_, slot) in enumerate(slots.iterrows()):
        dt_date = slot["_date_dt"].date()
        hr = int(slot["hour"])
        dow = slot["day_of_week"]

        if i % 50 == 0:
            print(f"  [backfill] Processing slot {i+1}/{total_slots}: {dt_date} hour {hr}")

        baseline_start = dt_date - timedelta(days=baseline_days)
        hist = all_data[
            (all_data["_date_dt"] >= pd.Timestamp(baseline_start))
            & (all_data["_date_dt"] < pd.Timestamp(dt_date))
        ]
        curr = all_data[
            (all_data["_date_dt"] == pd.Timestamp(dt_date))
            & (all_data["hour"] == hr)
        ]

        if hist.empty or curr.empty:
            continue

        anomalies = detect_anomalies(
            curr, hist, target_dow=dow, target_hour=hr, skip_cooldown=True,
        )

        for a in anomalies:
            records.append({
                "_date": dt_date,
                "hour": hr,
                "day_of_week": dow,
                "partner": a.partner,
                "level": a.level,
                "metric_key": a.metric_key,
                "current_rate": a.current_rate,
                "baseline_mean": a.baseline_mean,
                "baseline_std": a.baseline_std,
                "z_score": a.z_score,
                "pp_drop": a.pp_drop,
                "volume": a.volume,
            })

    raw = pd.DataFrame(records)
    if raw.empty:
        return raw
    return consolidate_incidents(raw)


# ── Incident consolidation ────────────────────────────────────────────────

MIN_CONSECUTIVE_HOURS = 2  # require 2+ consecutive anomaly hours to form an incident


def consolidate_incidents(raw_anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Group raw per-hour anomaly rows into *incidents*: contiguous runs of
    2+ consecutive hours for the same partner × metric × level.

    Each incident row keeps the worst (most negative) z-score hour's stats,
    plus the start/end hour and total duration.  Single-hour blips are dropped.
    """
    if raw_anomalies.empty:
        return raw_anomalies

    df = raw_anomalies.copy()
    df["_date"] = pd.to_datetime(df["_date"])
    df = df.sort_values(["partner", "level", "metric_key", "_date", "hour"])

    incidents: list[dict] = []
    group_keys = ["partner", "level", "metric_key"]

    for key, grp in df.groupby(group_keys, dropna=False):
        partner, level, metric_key = key
        grp = grp.sort_values(["_date", "hour"]).reset_index(drop=True)

        run_start = 0
        for i in range(1, len(grp) + 1):
            # Check if current row is consecutive with previous
            if i < len(grp):
                prev_date = grp.loc[i - 1, "_date"]
                prev_hour = grp.loc[i - 1, "hour"]
                curr_date = grp.loc[i, "_date"]
                curr_hour = grp.loc[i, "hour"]

                # Consecutive = same date and hour+1, OR next date hour 0 after hour 23
                is_consecutive = (
                    (curr_date == prev_date and curr_hour == prev_hour + 1)
                    or (curr_date == prev_date + pd.Timedelta(days=1)
                        and prev_hour == 23 and curr_hour == 0)
                )
                if is_consecutive:
                    continue

            # End of a run: rows run_start..i-1
            run = grp.iloc[run_start:i]
            run_len = len(run)

            if run_len >= MIN_CONSECUTIVE_HOURS:
                worst = run.loc[run["z_score"].idxmin()]
                incidents.append({
                    "_date": run.iloc[0]["_date"].date(),
                    "start_hour": int(run.iloc[0]["hour"]),
                    "end_hour": int(run.iloc[-1]["hour"]),
                    "duration_hours": run_len,
                    "day_of_week": run.iloc[0].get("day_of_week", ""),
                    "partner": partner,
                    "level": level,
                    "metric_key": metric_key,
                    "worst_rate": float(worst["current_rate"]),
                    "baseline_mean": float(worst["baseline_mean"]),
                    "baseline_std": float(worst["baseline_std"]),
                    "worst_z_score": float(worst["z_score"]),
                    "worst_pp_drop": float(worst["pp_drop"]),
                    "avg_pp_drop": float(run["pp_drop"].mean()),
                    "total_volume": int(run["volume"].sum()),
                })

            run_start = i

    return pd.DataFrame(incidents)


# ── Cooldown logic ───────────────────────────────────────────────────────

def _is_on_cooldown(partner: str, metric_key: str) -> bool:
    if not os.path.exists(COOLDOWN_LOG_PATH):
        return False
    try:
        with open(COOLDOWN_LOG_PATH) as f:
            log = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    key = f"{partner}::{metric_key}"
    last = log.get(key)
    if last is None:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return False
    return datetime.now() - last_dt < timedelta(hours=COOLDOWN_HOURS)


def record_cooldown(partner: str, metric_key: str) -> None:
    """Mark an alert as sent so duplicates are suppressed."""
    log: dict = {}
    if os.path.exists(COOLDOWN_LOG_PATH):
        try:
            with open(COOLDOWN_LOG_PATH) as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    log[f"{partner}::{metric_key}"] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(COOLDOWN_LOG_PATH), exist_ok=True)
    with open(COOLDOWN_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
