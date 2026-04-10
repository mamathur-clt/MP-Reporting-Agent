"""
Partner Health Monitor — metric aggregation.

Takes raw cart-session DataFrame, groups by partner, and computes
the five monitored metrics plus supporting context distributions.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from monitor.config import METRICS, PARTNER_TO_PARENT, MetricDef


# ── Helper ────────────────────────────────────────────────────────────────

def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise rate with 0 where denominator is 0."""
    return np.where(den > 0, num / den, np.nan)


def _map_parent(partner: str) -> str:
    """Map a partner name to its parent company, or return the partner itself."""
    return PARTNER_TO_PARENT.get(partner, partner)


# ── Per-partner metric computation ────────────────────────────────────────

def compute_partner_metrics(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    level: str = "partner",
) -> pd.DataFrame:
    """
    Aggregate the four rate-based metrics.

    level="partner"  → one row per individual provider (default)
    level="parent"   → aggregate child partners into parent companies
    level="both"     → return both partner-level AND parent-level rows

    Returns columns:
        partner, <group_cols>, metric_key, numerator_sum, denominator_sum,
        rate, volume, level
    """
    if group_cols is None:
        group_cols = []

    records: list[pd.DataFrame] = []

    for mdef in METRICS.values():
        pcol = mdef.partner_col
        if pcol not in df.columns:
            continue

        num_col = mdef.numerator
        den_col = mdef.denominator
        if num_col not in df.columns or den_col not in df.columns:
            continue

        grp_keys = [pcol] + [c for c in group_cols if c in df.columns]
        agg = (
            df[df[den_col] > 0]
            .groupby(grp_keys, dropna=False)
            .agg(
                numerator_sum=(num_col, "sum"),
                denominator_sum=(den_col, "sum"),
                volume=(den_col, "count"),
            )
            .reset_index()
        )
        agg["rate"] = _safe_rate(agg["numerator_sum"], agg["denominator_sum"])
        agg["metric_key"] = mdef.key
        agg["partner"] = agg[pcol].fillna("Unknown")
        agg = agg.drop(columns=[pcol])
        agg["level"] = "partner"

        if level in ("partner", "both"):
            records.append(agg)

        # Parent-company aggregation
        if level in ("parent", "both"):
            agg_copy = agg.copy()
            agg_copy["parent"] = agg_copy["partner"].apply(_map_parent)
            # Only aggregate where parent != partner (i.e. belongs to a group)
            has_parent = agg_copy[agg_copy["parent"] != agg_copy["partner"]]
            if not has_parent.empty:
                parent_grp = ["parent", "metric_key"] + [c for c in group_cols if c in has_parent.columns]
                parent_agg = (
                    has_parent
                    .groupby(parent_grp, dropna=False)
                    .agg(
                        numerator_sum=("numerator_sum", "sum"),
                        denominator_sum=("denominator_sum", "sum"),
                        volume=("volume", "sum"),
                    )
                    .reset_index()
                )
                parent_agg["rate"] = _safe_rate(parent_agg["numerator_sum"], parent_agg["denominator_sum"])
                parent_agg = parent_agg.rename(columns={"parent": "partner"})
                parent_agg["level"] = "parent"
                records.append(parent_agg)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    if "level" not in result.columns:
        result["level"] = "partner"
    return result


def get_child_partner_breakdown(
    df: pd.DataFrame,
    parent: str,
    metric_key: str,
) -> pd.DataFrame:
    """
    For a parent-company alert, return the per-child-partner breakdown
    so the alert can show which specific providers are affected.
    """
    from monitor.config import PARENT_COMPANIES
    children = PARENT_COMPANIES.get(parent, [])
    if not children:
        return pd.DataFrame()

    mdef = METRICS.get(metric_key)
    if mdef is None:
        return pd.DataFrame()

    pcol = mdef.partner_col
    if pcol not in df.columns:
        return pd.DataFrame()

    subset = df[df[pcol].isin(children)]
    if subset.empty:
        return pd.DataFrame()

    child_metrics = compute_partner_metrics(subset, level="partner")
    return child_metrics[child_metrics["metric_key"] == metric_key]


# ── Qual error breakdown ─────────────────────────────────────────────────

def compute_qual_error_breakdown(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the distribution of first_qual_error_reason per provider.
    Only includes rows where qual failed.
    """
    if group_cols is None:
        group_cols = []

    needed = {"qual_fail", "first_qual_error_reason", "midflow_provider"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    fails = df[df["qual_fail"] == 1].copy()
    if fails.empty:
        return pd.DataFrame()

    grp_keys = ["midflow_provider", "first_qual_error_reason"] + [
        c for c in group_cols if c in fails.columns
    ]

    agg = (
        fails.groupby(grp_keys, dropna=False)
        .size()
        .reset_index(name="count")
    )
    agg = agg.rename(columns={"midflow_provider": "partner"})
    agg["partner"] = agg["partner"].fillna("Unknown")
    agg["first_qual_error_reason"] = agg["first_qual_error_reason"].fillna("Unknown")

    # Share within each partner (and group)
    share_keys = ["partner"] + [c for c in group_cols if c in agg.columns]
    totals = agg.groupby(share_keys, dropna=False)["count"].transform("sum")
    agg["share"] = np.where(totals > 0, agg["count"] / totals, 0.0)

    return agg


# ── Credit score distribution ────────────────────────────────────────────

def compute_credit_score_distribution(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute credit score bucket distribution per provider, for rows that
    had a credit run.
    """
    if group_cols is None:
        group_cols = []

    needed = {"CreditScoreBucketCR", "first_run_provider_name", "has_credit_run"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    subset = df[df["has_credit_run"] == 1].copy()
    if subset.empty:
        return pd.DataFrame()

    grp_keys = ["first_run_provider_name", "CreditScoreBucketCR"] + [
        c for c in group_cols if c in subset.columns
    ]

    agg = (
        subset.groupby(grp_keys, dropna=False)
        .agg(
            count=("has_credit_run", "sum"),
            credit_pass_count=("credit_pass_flag", "sum"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={"first_run_provider_name": "partner"})
    agg["partner"] = agg["partner"].fillna("Unknown")
    agg["CreditScoreBucketCR"] = agg["CreditScoreBucketCR"].fillna("Unknown")
    agg["credit_fail_count"] = agg["count"] - agg["credit_pass_count"]

    return agg
