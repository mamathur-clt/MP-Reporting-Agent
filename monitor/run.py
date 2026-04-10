"""
Partner Health Monitor — runner.

Entry point for the monitoring pipeline.  Three modes:

    python -m monitor.run                 # one-shot check
    python -m monitor.run --daemon        # hourly APScheduler loop
    python -m monitor.run --backfill 2026-01-01  # recompute history
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from monitor.alerts import fire_alert
from monitor.anomaly import detect_anomalies, detect_hourly_anomalies
from monitor.config import BASELINE_DAYS, CHECK_INTERVAL_HOURS, DATA_START_DATE
from monitor.data import fetch_baseline, fetch_cart_data, fetch_recent_hours
from monitor.metrics import (
    compute_credit_score_distribution,
    compute_partner_metrics,
    compute_qual_error_breakdown,
)


# ── One-shot check ────────────────────────────────────────────────────────

def run_once() -> None:
    """Fetch recent data, compare to baseline, fire alerts for anomalies."""
    print(f"[monitor] One-shot run at {datetime.now():%Y-%m-%d %H:%M:%S}")

    print("[monitor] Fetching recent hour data …")
    current_df = fetch_recent_hours(n_hours=1)
    if current_df.empty:
        print("[monitor] No recent data — skipping.")
        return

    print("[monitor] Fetching baseline data …")
    history_df = fetch_baseline(days=BASELINE_DAYS)
    if history_df.empty:
        print("[monitor] No baseline data — skipping.")
        return

    now = datetime.now()
    dow = now.strftime("%A")
    hour = now.hour

    print(f"[monitor] Detecting anomalies (dow={dow}, hour={hour}) …")
    anomalies = detect_anomalies(current_df, history_df, target_dow=dow, target_hour=hour)

    if not anomalies:
        print("[monitor] No anomalies detected. ✓")
        return

    print(f"[monitor] {len(anomalies)} anomalies detected — preparing alerts …")

    # Pre-compute context distributions for alert enrichment
    history_metrics = compute_partner_metrics(history_df, group_cols=["_date"])
    credit_dist = compute_credit_score_distribution(current_df)
    qual_errors = compute_qual_error_breakdown(current_df)

    for a in anomalies:
        level_tag = f" [{a.level}]" if a.level == "parent" else ""
        print(f"  → {a.partner}{level_tag} | {a.metric_key}: {a.current_rate:.1%} "
              f"(baseline {a.baseline_mean:.1%}, z={a.z_score:+.1f})")
        fire_alert(
            a,
            history_metrics=history_metrics,
            credit_dist=credit_dist,
            qual_errors=qual_errors,
            current_df=current_df,
        )

    print("[monitor] Done.")


# ── Daemon mode ───────────────────────────────────────────────────────────

def run_daemon() -> None:
    """Run the monitor on a recurring schedule using APScheduler."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        print("[monitor] APScheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler()
    scheduler.add_job(run_once, "interval", hours=CHECK_INTERVAL_HOURS, next_run_time=datetime.now())
    print(f"[monitor] Daemon started — running every {CHECK_INTERVAL_HOURS}h. Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[monitor] Daemon stopped.")


# ── Backfill mode ─────────────────────────────────────────────────────────

def run_backfill(start_date_str: str) -> None:
    """
    Recompute anomalies for every day from *start_date_str* to today.
    Results are saved to monitor/backfill_results.csv for the dashboard.
    """
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_dt = date.today()

    print(f"[monitor] Backfill from {start_dt} to {end_dt}")
    print("[monitor] Fetching full date range from Databricks …")
    all_data = fetch_cart_data(str(start_dt), str(end_dt))
    if all_data.empty:
        print("[monitor] No data returned — aborting backfill.")
        return

    print(f"[monitor] {len(all_data)} rows loaded. Running hourly anomaly detection …")
    results = detect_hourly_anomalies(all_data, baseline_days=BASELINE_DAYS)

    if results.empty:
        print("[monitor] No anomalies found in backfill window.")
        return

    import os
    from monitor.config import REPO_ROOT

    out_path = os.path.join(REPO_ROOT, "monitor", "backfill_results.csv")
    results.to_csv(out_path, index=False)
    print(f"[monitor] {len(results)} anomaly records written to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Partner Health Monitor")
    parser.add_argument("--daemon", action="store_true", help="Run on a recurring hourly schedule")
    parser.add_argument("--backfill", type=str, default=None, metavar="YYYY-MM-DD",
                        help=f"Backfill anomaly detection from this date to today (default: {DATA_START_DATE})")
    args = parser.parse_args()

    if args.daemon:
        run_daemon()
    elif args.backfill is not None:
        run_backfill(args.backfill or DATA_START_DATE)
    else:
        run_once()


if __name__ == "__main__":
    main()
