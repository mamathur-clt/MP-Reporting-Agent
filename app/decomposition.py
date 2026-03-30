"""
Decomposition engine for explaining period-over-period KPI movement.

Uses a **mix-shift / rate-change** framework:

For a KPI = SUM(numerator) / SUM(denominator), when sliced by a dimension
with segments s1, s2, …, the overall rate R = Σ (w_s × r_s) where:
  w_s = share of denominator in segment s
  r_s = KPI rate within segment s

Period-over-period change ΔR decomposes into per-segment contributions:
  contribution_s = (Δw_s × r̄_s) + (Δr_s × w̄_s)
                   ~~~~~~~~~~~     ~~~~~~~~~~~
                   mix effect       rate effect

where r̄ = avg of current/prior rate, w̄ = avg of current/prior weight.

This is the classic Shapley-value-consistent additive decomposition that
sums exactly to the total ΔR.  It is interpretable: "segment X contributed
+0.3pp because its share grew and its rate improved."

For initiatives, we use a counterfactual approach:
  "What would the KPI have been if model sessions had performed at
   holdout rates?"  The gap is the initiative's contribution.
"""

import pandas as pd
import numpy as np
from app.config import KPIS, DRIVER_DIMENSIONS, INITIATIVE_COLUMNS


def _safe_div(a, b):
    return a / b if b != 0 else 0.0


def decompose_by_dimension(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame,
    kpi_key: str,
    dimension: str,
) -> pd.DataFrame:
    """
    Decompose ΔR into per-segment contributions along *dimension*.
    Returns DataFrame with columns:
      segment, curr_weight, prior_weight, curr_rate, prior_rate,
      mix_effect, rate_effect, total_contribution
    """
    kpi = KPIS[kpi_key]
    num_col, den_col = kpi.numerator, kpi.denominator

    def _agg(df):
        g = df.groupby(dimension, dropna=False).agg(
            num=(num_col, "sum"), den=(den_col, "sum")
        ).reset_index()
        total_den = g["den"].sum()
        g["weight"] = g["den"] / total_den if total_den > 0 else 0.0
        g["rate"] = np.where(g["den"] > 0, g["num"] / g["den"], 0.0)
        return g

    curr = _agg(df_current).rename(columns={
        "weight": "curr_weight", "rate": "curr_rate",
        "num": "curr_num", "den": "curr_den",
    })
    prior = _agg(df_prior).rename(columns={
        "weight": "prior_weight", "rate": "prior_rate",
        "num": "prior_num", "den": "prior_den",
    })

    merged = pd.merge(
        curr, prior, on=dimension, how="outer", suffixes=("", "_p")
    ).fillna(0)

    merged["delta_weight"] = merged["curr_weight"] - merged["prior_weight"]
    merged["delta_rate"] = merged["curr_rate"] - merged["prior_rate"]
    merged["avg_rate"] = (merged["curr_rate"] + merged["prior_rate"]) / 2
    merged["avg_weight"] = (merged["curr_weight"] + merged["prior_weight"]) / 2

    merged["mix_effect"] = merged["delta_weight"] * merged["avg_rate"]
    merged["rate_effect"] = merged["delta_rate"] * merged["avg_weight"]
    merged["total_contribution"] = merged["mix_effect"] + merged["rate_effect"]

    merged = merged.rename(columns={dimension: "segment"})
    keep = [
        "segment", "curr_den", "prior_den",
        "curr_weight", "prior_weight", "curr_rate", "prior_rate",
        "mix_effect", "rate_effect", "total_contribution",
    ]
    return merged[[c for c in keep if c in merged.columns]].sort_values(
        "total_contribution", key=abs, ascending=False
    )


def decompose_all_dimensions(
    df_current: pd.DataFrame,
    df_prior: pd.DataFrame,
    kpi_key: str,
    dimensions: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run decomposition across multiple dimensions, return dict of results."""
    dims = dimensions or DRIVER_DIMENSIONS
    results = {}
    for dim in dims:
        if dim in df_current.columns and dim in df_prior.columns:
            results[dim] = decompose_by_dimension(
                df_current, df_prior, kpi_key, dim
            )
    return results


def rank_top_drivers(
    decomp_results: dict[str, pd.DataFrame],
    n: int = 10,
) -> pd.DataFrame:
    """
    Flatten all dimension decompositions and rank by absolute contribution.
    Returns the top N positive and negative drivers combined.
    """
    all_rows = []
    for dim, df in decomp_results.items():
        sub = df[["segment", "mix_effect", "rate_effect", "total_contribution"]].copy()
        sub["dimension"] = dim
        all_rows.append(sub)
    if not all_rows:
        return pd.DataFrame()
    combined = pd.concat(all_rows, ignore_index=True)
    combined["abs_contribution"] = combined["total_contribution"].abs()
    combined = combined.sort_values("abs_contribution", ascending=False)
    top_positive = combined[combined["total_contribution"] > 0].head(n)
    top_negative = combined[combined["total_contribution"] < 0].head(n)
    return pd.concat([top_positive, top_negative]).sort_values(
        "total_contribution", ascending=True
    )


# ---------------------------------------------------------------------------
# Initiative counterfactual analysis
# ---------------------------------------------------------------------------

def _initiative_label(row: pd.Series) -> str:
    """
    Assign an initiative label to each session based on flags.

    lp_experience has three values for LP agent sessions:
      'Holdout' — LP holdout (shared control for LP Model and FMP)
      'Model'   — LP JO model (explore/exploit)
      'FMP'     — Find My Plan (intuitive_explore)
    """
    lp_exp = str(row.get("lp_experience", "")).strip()
    has_lp = row.get("is_lp", 0) == 1 or row.get("grid_lp", 0) == 1
    has_grid = row.get("is_grid", 0) == 1 or row.get("grid_lp", 0) == 1
    holdout = row.get("isHoldout", "")

    if lp_exp == "FMP":
        if has_grid:
            return "FMP + Grid JO"
        return "FMP"

    if has_lp and has_grid:
        return f"LP+Grid JO {'Holdout' if holdout == 'Holdout' else 'Model'}"
    if has_lp:
        return f"LP JO {'Holdout' if holdout == 'Holdout' else 'Model'}"
    if has_grid:
        return f"Grid JO {'Holdout' if holdout == 'Holdout' else 'Model'}"
    return "No Initiative"


def analyze_initiatives(
    df: pd.DataFrame,
    kpi_key: str,
) -> pd.DataFrame:
    """
    For a single period, compute KPI by initiative group and compare
    model vs holdout for each initiative.

    Returns DataFrame with initiative_group, rate, volume, and
    counterfactual_delta (how much worse/better vs holdout).
    """
    kpi = KPIS[kpi_key]
    num_col, den_col = kpi.numerator, kpi.denominator

    df = df.copy()
    df["_initiative"] = df.apply(_initiative_label, axis=1)

    agg = df.groupby("_initiative").agg(
        num=(num_col, "sum"), den=(den_col, "sum"), sessions=("session", "sum")
    ).reset_index()
    agg["rate"] = np.where(agg["den"] > 0, agg["num"] / agg["den"], 0.0)

    # Compute holdout rate for counterfactual comparisons.
    # FMP shares the LP holdout as its control (same agent 3313).
    holdout_rates = {}
    for init_type in ["LP+Grid JO", "LP JO", "Grid JO"]:
        holdout_row = agg[agg["_initiative"] == f"{init_type} Holdout"]
        if not holdout_row.empty:
            holdout_rates[init_type] = holdout_row.iloc[0]["rate"]

    # FMP control = LP+Grid JO Holdout if present, else LP JO Holdout
    for fallback in ["LP+Grid JO Holdout", "LP JO Holdout"]:
        h_row = agg[agg["_initiative"] == fallback]
        if not h_row.empty:
            holdout_rates["FMP"] = h_row.iloc[0]["rate"]
            holdout_rates["FMP + Grid JO"] = h_row.iloc[0]["rate"]
            break

    rows = []
    for _, row in agg.iterrows():
        init = row["_initiative"]
        counterfactual_delta = 0.0

        if init in ("FMP", "FMP + Grid JO"):
            h_rate = holdout_rates.get("FMP")
            if h_rate is not None:
                counterfactual_delta = row["rate"] - h_rate
        else:
            for init_type, h_rate in holdout_rates.items():
                if init == f"{init_type} Model":
                    counterfactual_delta = row["rate"] - h_rate
                    break

        rows.append({
            "initiative": init,
            "sessions": int(row["sessions"]),
            "denominator": int(row["den"]),
            "numerator": int(row["num"]),
            "rate": row["rate"],
            "counterfactual_vs_holdout": counterfactual_delta,
        })

    return pd.DataFrame(rows).sort_values("sessions", ascending=False)


def compute_initiative_impact(
    df: pd.DataFrame,
    kpi_key: str,
) -> pd.DataFrame:
    """
    Compute the scaled impact of each initiative on the all-in KPI.

    For each model/FMP group, the logic is:
      1. Compute model_rate and holdout_rate
      2. lift = model_rate - holdout_rate
      3. model_share = model_denominator / total_denominator
      4. scaled_impact = lift × model_share
         (how much the initiative's lift adds to the all-in KPI
          vs a world where those sessions performed at holdout rates)
      5. Sum across all initiatives = total initiative contribution

    This answers: "initiatives contributed +X pp to the overall KPI
    this period" and lets you compare current vs prior.
    """
    kpi = KPIS[kpi_key]
    num_col, den_col = kpi.numerator, kpi.denominator

    df = df.copy()
    df["_initiative"] = df.apply(_initiative_label, axis=1)

    agg = df.groupby("_initiative").agg(
        num=(num_col, "sum"), den=(den_col, "sum"), sessions=("session", "sum")
    ).reset_index()
    agg["rate"] = np.where(agg["den"] > 0, agg["num"] / agg["den"], 0.0)

    total_den = agg["den"].sum()

    # Find holdout rates
    holdout_rates = {}
    for init_type in ["LP+Grid JO", "LP JO", "Grid JO"]:
        h_row = agg[agg["_initiative"] == f"{init_type} Holdout"]
        if not h_row.empty:
            holdout_rates[init_type] = h_row.iloc[0]["rate"]

    for fallback in ["LP+Grid JO Holdout", "LP JO Holdout"]:
        h_row = agg[agg["_initiative"] == fallback]
        if not h_row.empty:
            holdout_rates["FMP"] = h_row.iloc[0]["rate"]
            holdout_rates["FMP + Grid JO"] = h_row.iloc[0]["rate"]
            break

    # Map each model/FMP group to its holdout type
    model_to_holdout = {
        "LP+Grid JO Model": "LP+Grid JO",
        "LP JO Model": "LP JO",
        "Grid JO Model": "Grid JO",
        "FMP": "FMP",
        "FMP + Grid JO": "FMP + Grid JO",
    }

    rows = []
    for _, row in agg.iterrows():
        init = row["_initiative"]
        holdout_key = model_to_holdout.get(init)
        if holdout_key is None:
            continue

        h_rate = holdout_rates.get(holdout_key)
        if h_rate is None:
            continue

        lift = row["rate"] - h_rate
        share = row["den"] / total_den if total_den > 0 else 0.0
        scaled_impact = lift * share

        rows.append({
            "initiative": init,
            "model_rate": row["rate"],
            "holdout_rate": h_rate,
            "lift": lift,
            "model_sessions": int(row["sessions"]),
            "model_share": share,
            "scaled_impact_on_kpi": scaled_impact,
        })

    return pd.DataFrame(rows).sort_values("scaled_impact_on_kpi", ascending=False)
