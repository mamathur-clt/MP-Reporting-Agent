"""
APG&E Credit Threshold Change Analysis
=======================================
Event: APG&E raised its credit threshold from 500 → 600 on 3/23 morning.
Impact: Sessions with credit scores 500-599 that previously passed APG&E
        credit check now fail, directly hitting Conversion After Credit
        and flowing through to Cart VC and all-in VC.

This script queries the session-level data, computes the impact, builds
a counterfactual, and produces a written TLDR.

Run:  python adhoc_analyses/apge_credit_threshold/analysis.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import certifi
import pandas as pd
import numpy as np
from datetime import date
from dotenv import load_dotenv
from databricks import sql as databricks_sql

load_dotenv(override=True)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

_HOST = os.getenv("DATABRICKS_HOST", "")
_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")

DEFAULT_CHANNELS = ["Paid Search", "Direct", "Organic", "pMax"]

# ── Dates ──────────────────────────────────────────────────────────────────
# Pre-change week:  Mon 3/16 → Sun 3/22
# Post-change week: Mon 3/23 → Sun 3/29 (or today if mid-week)
PRE_START, PRE_END = date(2026, 3, 16), date(2026, 3, 22)
POST_START, POST_END = date(2026, 3, 23), date(2026, 3, 29)
QUERY_START, QUERY_END = PRE_START, POST_END


def _get_connection():
    return databricks_sql.connect(
        server_hostname=_HOST.replace("https://", "").strip("/"),
        http_path=_HTTP_PATH,
        access_token=_TOKEN,
    )


def _build_query(start_date: str, end_date: str) -> str:
    query_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "session_level_query"
    )
    with open(query_path) as f:
        raw = f.read()
    raw = raw.replace("%sql", "", 1).strip()
    raw = raw.replace(
        "('2026-01-01')::date AS start_date",
        f"('{start_date}')::date AS start_date",
    )
    raw = raw.replace(
        "(CURRENT_DATE)::date AS end_date",
        f"('{end_date}')::date AS end_date",
    )
    return raw


def fetch_data() -> pd.DataFrame:
    print(f"Querying Databricks: {QUERY_START} → {QUERY_END} ...")
    query = _build_query(str(QUERY_START), str(QUERY_END))
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=cols)

    if "session_start_date_est" in df.columns:
        df["session_start_date_est"] = pd.to_datetime(
            df["session_start_date_est"]
        ).dt.date

    for col in ["cart_order", "phone_order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_order"] = df.get("cart_order", 0) + df.get("phone_order", 0)

    flag_cols = [
        "session", "zip_entry", "has_cart", "cart_order", "cart_ssn_done",
        "gross_call", "queue_call", "phone_order", "cart_credit_fail",
        "cart_provider_pass", "cart_volt_fail", "cart_qual_fail", "total_order",
    ]
    for col in flag_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "first_run_credit_score" in df.columns:
        df["first_run_credit_score"] = pd.to_numeric(
            df["first_run_credit_score"], errors="coerce"
        )

    return df


def safe_rate(num, den):
    return num / den if den > 0 else 0.0


def pct_change(curr, prior):
    if prior == 0:
        return float("nan")
    return (curr - prior) / prior


def fmt_pct(val):
    return f"{val * 100:.2f}%"


def fmt_pct_ch(val):
    return f"{val * 100:+.1f}%"


def credit_band(score):
    if pd.isna(score):
        return "No Credit Run"
    s = int(score)
    if s < 500:
        return "<500"
    elif s < 600:
        return "500-599"
    elif s < 700:
        return "600-699"
    elif s < 800:
        return "700-799"
    else:
        return "800+"


def run_analysis(df: pd.DataFrame):
    # Filter to core channels
    df = df[df["marketing_channel"].isin(DEFAULT_CHANNELS)].copy()

    df["period"] = df["session_start_date_est"].apply(
        lambda d: "Pre (3/16-3/22)" if d <= PRE_END else "Post (3/23-3/29)"
    )
    df["credit_band"] = df["first_run_credit_score"].apply(credit_band)
    df["is_apge"] = df["first_run_provider_name"].fillna("").str.upper().str.contains("APG")

    # ── 1. Overall funnel impact ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("APG&E CREDIT THRESHOLD ANALYSIS: 500 → 600 on 3/23")
    print("=" * 80)

    print("\n── 1. IMPACT TO PERFORMANCE ──────────────────────────────────\n")

    kpis = [
        ("Cart RR",              "has_cart",       "session"),
        ("Conversion After Credit", "cart_order",  "cart_ssn_done"),
        ("Cart Conversion",      "cart_order",     "has_cart"),
        ("Cart VC",              "cart_order",     "session"),
        ("VC (Total)",           "total_order",    "session"),
    ]

    # Overall WoW
    pre = df[df["period"] == "Pre (3/16-3/22)"]
    post = df[df["period"] == "Post (3/23-3/29)"]

    print("All-in KPI movement (WoW):")
    print(f"  {'KPI':<28} {'Pre':>10} {'Post':>10} {'Change':>10}")
    print("  " + "-" * 60)
    for label, num_col, den_col in kpis:
        r_pre = safe_rate(pre[num_col].sum(), pre[den_col].sum())
        r_post = safe_rate(post[num_col].sum(), post[den_col].sum())
        ch = pct_change(r_post, r_pre)
        print(f"  {label:<28} {fmt_pct(r_pre):>10} {fmt_pct(r_post):>10} {fmt_pct_ch(ch):>10}")

    # ── 2. Isolate the affected population ──────────────────────────────
    print("\n\n── APG&E Credit Check: 500-599 Band ─────────────────────────\n")

    affected_pre = pre[
        (pre["credit_band"] == "500-599") & (pre["is_apge"]) & (pre["cart_ssn_done"] == 1)
    ]
    affected_post = post[
        (post["credit_band"] == "500-599") & (post["is_apge"]) & (post["cart_ssn_done"] == 1)
    ]

    n_pre = len(affected_pre)
    n_post = len(affected_post)
    orders_pre = affected_pre["cart_order"].sum()
    orders_post = affected_post["cart_order"].sum()
    conv_pre = safe_rate(orders_pre, n_pre)
    conv_post = safe_rate(orders_post, n_post)

    print(f"  APG&E first-run, credit score 500-599, SSN-submitted sessions:")
    print(f"    Pre week:  {int(n_pre):,} sessions → {int(orders_pre):,} orders → Conv After Credit = {fmt_pct(conv_pre)}")
    print(f"    Post week: {int(n_post):,} sessions → {int(orders_post):,} orders → Conv After Credit = {fmt_pct(conv_post)}")
    if n_pre > 0 and n_post > 0:
        print(f"    Change: {fmt_pct_ch(pct_change(conv_post, conv_pre))}")

    # Also show credit fail rate for this band
    cf_pre = safe_rate(affected_pre["cart_credit_fail"].sum(), n_pre)
    cf_post = safe_rate(affected_post["cart_credit_fail"].sum(), n_post)
    print(f"\n  Credit fail rate for this band:")
    print(f"    Pre:  {fmt_pct(cf_pre)}")
    print(f"    Post: {fmt_pct(cf_post)}")

    # ── 3. Conversion After Credit by credit band and provider ─────────
    print("\n\n── Conversion After Credit by Credit Band (APG&E first-run) ──\n")

    apge_ssn = df[(df["is_apge"]) & (df["cart_ssn_done"] == 1)]

    band_data = apge_ssn.groupby(["period", "credit_band"]).agg(
        ssn_sessions=("cart_ssn_done", "sum"),
        orders=("cart_order", "sum"),
    ).reset_index()
    band_data["conv_after_credit"] = band_data.apply(
        lambda r: safe_rate(r["orders"], r["ssn_sessions"]), axis=1
    )

    print(f"  {'Credit Band':<15} {'Period':<20} {'SSN Sessions':>14} {'Orders':>10} {'Conv After Credit':>18}")
    print("  " + "-" * 80)
    for band in ["<500", "500-599", "600-699", "700-799", "800+"]:
        rows = band_data[band_data["credit_band"] == band].sort_values("period")
        for _, r in rows.iterrows():
            print(f"  {r['credit_band']:<15} {r['period']:<20} {int(r['ssn_sessions']):>14,} {int(r['orders']):>10,} {fmt_pct(r['conv_after_credit']):>18}")

    # ── 4. Size the 500-599 population relative to all-in ──────────────
    print("\n\n── Size of affected population ───────────────────────────────\n")

    total_ssn_pre = pre["cart_ssn_done"].sum()
    total_ssn_post = post["cart_ssn_done"].sum()
    print(f"  Total SSN-submitted sessions pre week:  {int(total_ssn_pre):,}")
    print(f"  Total SSN-submitted sessions post week: {int(total_ssn_post):,}")
    print(f"  APG&E 500-599 pre week:  {int(n_pre):,} ({safe_rate(n_pre, total_ssn_pre) * 100:.1f}% of SSN sessions)")
    print(f"  APG&E 500-599 post week: {int(n_post):,} ({safe_rate(n_post, total_ssn_post) * 100:.1f}% of SSN sessions)")

    # ── 5. Counterfactual Analysis ──────────────────────────────────────
    print("\n\n── 2. COUNTERFACTUAL ANALYSIS ────────────────────────────────\n")
    print("  Question: How much of the all-in VC/Cart VC decline is explained")
    print("  by the APG&E credit threshold change?\n")

    # Counterfactual: if the 500-599 APG&E band in the post period had
    # maintained their pre-period conversion-after-credit rate, how many
    # more orders would we have?

    counterfactual_orders = 0.0
    if conv_pre > conv_post and n_post > 0:
        counterfactual_orders = (conv_pre - conv_post) * n_post

    total_orders_post = post["total_order"].sum()
    total_sessions_post = post["session"].sum()
    vc_post = safe_rate(total_orders_post, total_sessions_post)

    vc_counterfactual = safe_rate(total_orders_post + counterfactual_orders, total_sessions_post)

    cart_orders_post = post["cart_order"].sum()
    cart_vc_post = safe_rate(cart_orders_post, total_sessions_post)
    cart_vc_counterfactual = safe_rate(cart_orders_post + counterfactual_orders, total_sessions_post)

    print(f"  If 500-599 APG&E sessions had maintained pre-week conversion:")
    print(f"    Lost orders (estimated): {counterfactual_orders:,.1f}")
    print(f"")
    print(f"  Cart VC impact:")
    print(f"    Actual post-week Cart VC:         {fmt_pct(cart_vc_post)}")
    print(f"    Counterfactual Cart VC:            {fmt_pct(cart_vc_counterfactual)}")
    print(f"    APG&E threshold cost to Cart VC:   {(cart_vc_counterfactual - cart_vc_post) * 100:+.3f}pp")
    print(f"")
    print(f"  All-in VC impact:")
    print(f"    Actual post-week VC:               {fmt_pct(vc_post)}")
    print(f"    Counterfactual VC:                 {fmt_pct(vc_counterfactual)}")
    print(f"    APG&E threshold cost to VC:        {(vc_counterfactual - vc_post) * 100:+.3f}pp")

    # What % of total VC WoW decline does this explain?
    total_orders_pre = pre["total_order"].sum()
    total_sessions_pre = pre["session"].sum()
    vc_pre = safe_rate(total_orders_pre, total_sessions_pre)
    vc_delta = vc_post - vc_pre
    apge_cost = vc_counterfactual - vc_post

    if vc_delta != 0:
        pct_explained = (apge_cost / abs(vc_delta)) * 100
        print(f"\n  Total VC WoW change: {vc_delta * 100:+.3f}pp")
        print(f"  APG&E threshold explains: {pct_explained:.1f}% of the total VC movement")

    # ── 6. Also check: did APG&E volume shift? ─────────────────────────
    print("\n\n── Provider mix check ────────────────────────────────────────\n")

    provider_pre = pre[pre["cart_ssn_done"] == 1].groupby("first_run_provider_name").agg(
        sessions=("cart_ssn_done", "sum"), orders=("cart_order", "sum")
    ).reset_index()
    provider_post = post[post["cart_ssn_done"] == 1].groupby("first_run_provider_name").agg(
        sessions=("cart_ssn_done", "sum"), orders=("cart_order", "sum")
    ).reset_index()

    providers = pd.merge(
        provider_pre.rename(columns={"sessions": "pre_sessions", "orders": "pre_orders"}),
        provider_post.rename(columns={"sessions": "post_sessions", "orders": "post_orders"}),
        on="first_run_provider_name", how="outer"
    ).fillna(0)
    providers["pre_conv"] = providers.apply(lambda r: safe_rate(r["pre_orders"], r["pre_sessions"]), axis=1)
    providers["post_conv"] = providers.apply(lambda r: safe_rate(r["post_orders"], r["post_sessions"]), axis=1)
    providers = providers.sort_values("post_sessions", ascending=False)

    print(f"  {'Provider':<30} {'Pre SSN':>10} {'Post SSN':>10} {'Pre Conv':>10} {'Post Conv':>10}")
    print("  " + "-" * 75)
    for _, r in providers.head(10).iterrows():
        print(
            f"  {str(r['first_run_provider_name']):<30} "
            f"{int(r['pre_sessions']):>10,} {int(r['post_sessions']):>10,} "
            f"{fmt_pct(r['pre_conv']):>10} {fmt_pct(r['post_conv']):>10}"
        )

    # ── 7. TLDR ─────────────────────────────────────────────────────────
    print("\n\n── 3. TLDR ──────────────────────────────────────────────────\n")

    direction = "declined" if conv_post < conv_pre else "improved"
    conv_ch = pct_change(conv_post, conv_pre)

    tldr = f"""  APG&E raised its credit threshold from 500 to 600 on 3/23.
  
  The 500-599 credit band for APG&E first-run credit checks saw
  Conversion After Credit {direction} {fmt_pct_ch(conv_ch)} WoW
  (from {fmt_pct(conv_pre)} to {fmt_pct(conv_post)}).
  
  This population represents ~{safe_rate(n_post, total_ssn_post) * 100:.1f}% of all SSN-submitted
  sessions. The estimated cost is ~{counterfactual_orders:,.0f} lost cart orders in
  the post week.
  
  Impact on all-in KPIs:
    - Cart VC: {(cart_vc_counterfactual - cart_vc_post) * 100:+.3f}pp drag
    - VC:      {(vc_counterfactual - vc_post) * 100:+.3f}pp drag"""

    if vc_delta != 0:
        tldr += f"""
  
  This explains approximately {pct_explained:.0f}% of the total WoW VC
  movement ({vc_delta * 100:+.3f}pp)."""

    tldr += """
  
  Sessions in this credit band that were previously converting through
  APG&E are now failing credit check. Some may re-run with another
  provider, but the immediate hit to Conversion After Credit is
  mechanical — these sessions can no longer qualify for APG&E plans."""

    print(tldr)
    print()

    return {
        "conv_pre": conv_pre,
        "conv_post": conv_post,
        "n_pre": n_pre,
        "n_post": n_post,
        "counterfactual_orders": counterfactual_orders,
        "apge_cost_cart_vc_pp": (cart_vc_counterfactual - cart_vc_post) * 100,
        "apge_cost_vc_pp": (vc_counterfactual - vc_post) * 100,
    }


if __name__ == "__main__":
    df = fetch_data()
    print(f"Loaded {len(df):,} sessions")
    results = run_analysis(df)
