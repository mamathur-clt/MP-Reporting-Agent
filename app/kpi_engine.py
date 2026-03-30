"""
KPI computation engine.

Computes KPIs from session-level DataFrames using the formulas defined in
full_funnel.docx. All KPIs follow the pattern: SUM(numerator) / SUM(denominator).
"""

import pandas as pd
from app.config import KPIS, KPIDef


def compute_kpi(df: pd.DataFrame, kpi_key: str) -> dict:
    """
    Compute a single KPI from a DataFrame slice.
    Returns dict with 'numerator_sum', 'denominator_sum', 'rate'.
    """
    kpi = KPIS[kpi_key]
    num = df[kpi.numerator].sum() if kpi.numerator in df.columns else 0
    den = df[kpi.denominator].sum() if kpi.denominator in df.columns else 0
    rate = num / den if den > 0 else 0.0
    return {"numerator_sum": num, "denominator_sum": den, "rate": rate}


def compute_all_kpis(df: pd.DataFrame) -> dict[str, dict]:
    """Compute all registered KPIs for a DataFrame."""
    return {k: compute_kpi(df, k) for k in KPIS}


def compute_kpi_summary(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame,
    kpi_key: str,
) -> dict:
    """
    Compute current, prior, delta, and percent change for one KPI.
    """
    curr = compute_kpi(df_current, kpi_key)
    prior = compute_kpi(df_prior, kpi_key)
    delta = curr["rate"] - prior["rate"]
    pct_change = delta / prior["rate"] if prior["rate"] != 0 else 0.0
    return {
        "kpi": kpi_key,
        "current_rate": curr["rate"],
        "prior_rate": prior["rate"],
        "delta": delta,
        "pct_change": pct_change,
        "current_numerator": curr["numerator_sum"],
        "current_denominator": curr["denominator_sum"],
        "prior_numerator": prior["numerator_sum"],
        "prior_denominator": prior["denominator_sum"],
    }


def compute_funnel_table(
    df_current: pd.DataFrame, df_prior: pd.DataFrame
) -> pd.DataFrame:
    """Build a full funnel summary table across all KPIs."""
    rows = []
    for k in KPIS:
        s = compute_kpi_summary(df_current, df_prior, k)
        rows.append(s)
    return pd.DataFrame(rows)
