"""
Generates a business-readable written summary of KPI movement and
its key drivers using OpenAI, suitable for weekly performance discussion.

Falls back to a structured template if the API call fails.
"""

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from app.config import KPIS, DIMENSION_DISPLAY_NAMES

load_dotenv(override=True)

_client = None
_client_key = None


def _get_client() -> OpenAI:
    global _client, _client_key
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "No OpenAI API key configured. Paste one into the sidebar to "
            "enable AI-generated summaries."
        )
    if _client is None or key != _client_key:
        _client = OpenAI(api_key=key)
        _client_key = key
    return _client


def _fmt_rate(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def _fmt_pp(delta: float) -> str:
    """Format a rate delta as percentage points (e.g. +0.02pp)."""
    pp = delta * 100
    return f"{pp:+.2f}pp"


def _dim_label(dim: str) -> str:
    return DIMENSION_DISPLAY_NAMES.get(dim, dim.replace("_", " ").title())


def _pct_change(delta: float, base: float) -> str:
    if base == 0:
        return "N/A"
    pct = (delta / base) * 100
    return f"{pct:+.1f}%"


def _driver_pct_of_total(driver_contribution: float, total_delta: float) -> str:
    """What % of the total KPI movement this driver explains."""
    if total_delta == 0:
        return "N/A"
    pct = (driver_contribution / total_delta) * 100
    return f"{abs(pct):.0f}%"


def _build_data_context(
    summary: dict,
    top_drivers: pd.DataFrame,
    initiative_df: pd.DataFrame | None = None,
    initiative_impact: pd.DataFrame | None = None,
    initiative_impact_prior: pd.DataFrame | None = None,
) -> str:
    """Build a structured data payload for the LLM prompt."""
    kpi_def = KPIS[summary["kpi"]]
    lines = []

    pct_ch = _pct_change(summary["delta"], summary["prior_rate"])

    lines.append(f"KPI: {kpi_def.name} ({summary['kpi']})")
    lines.append(f"Current period rate: {_fmt_rate(summary['current_rate'])}")
    lines.append(f"Prior period rate: {_fmt_rate(summary['prior_rate'])}")
    lines.append(f"% change: {pct_ch}")
    lines.append(f"Change in pp: {_fmt_pp(summary['delta'])}")
    lines.append(
        f"Current volume: {int(summary['current_numerator']):,} / "
        f"{int(summary['current_denominator']):,}"
    )
    lines.append(
        f"Prior volume: {int(summary['prior_numerator']):,} / "
        f"{int(summary['prior_denominator']):,}"
    )

    total_delta = summary["delta"]

    if not top_drivers.empty:
        lines.append("\nTop drivers of change (mix-shift / rate-change decomposition):")
        lines.append("  (% of total = what share of the overall KPI movement this driver explains)")
        for _, row in top_drivers.iterrows():
            pp = row["total_contribution"] * 100
            mix_pp = row["mix_effect"] * 100
            rate_pp = row["rate_effect"] * 100
            dim_name = _dim_label(row["dimension"])
            pct_of_total = _driver_pct_of_total(row["total_contribution"], total_delta)
            lines.append(
                f"  {dim_name} = {row['segment']}: "
                f"total {pp:+.2f}pp ({pct_of_total} of total change) "
                f"(mix {mix_pp:+.2f}pp, rate {rate_pp:+.2f}pp)"
            )

    if initiative_impact is not None and not initiative_impact.empty:
        lines.append("\nInitiative impact on all-in KPI (current period):")
        lines.append("  (lift = model rate - holdout rate; scaled impact = lift × share of traffic)")
        total_impact = initiative_impact["scaled_impact_on_kpi"].sum() * 100
        lines.append(f"  Total initiative contribution: {total_impact:+.3f}pp")
        for _, row in initiative_impact.iterrows():
            lines.append(
                f"  {row['initiative']}: model {_fmt_rate(row['model_rate'])}, "
                f"holdout {_fmt_rate(row['holdout_rate'])}, "
                f"lift {row['lift'] * 100:+.2f}pp, "
                f"share {row['model_share'] * 100:.1f}%, "
                f"scaled impact {row['scaled_impact_on_kpi'] * 100:+.3f}pp"
            )

    if initiative_impact_prior is not None and not initiative_impact_prior.empty:
        total_prior = initiative_impact_prior["scaled_impact_on_kpi"].sum() * 100
        lines.append(f"\n  Prior period total initiative contribution: {total_prior:+.3f}pp")
        if initiative_impact is not None:
            total_curr = initiative_impact["scaled_impact_on_kpi"].sum() * 100
            lines.append(f"  Change in initiative contribution: {total_curr - total_prior:+.3f}pp")

    return "\n".join(lines)


_BUSINESS_CONTEXT = """## Business Model

Energy marketplace connecting Texas consumers with electricity providers. Revenue model: commission per enrollment from electricity providers.

Four sites:
- **ChooseTexasPower.org (CTXP)** — largest by volume
- **SaveOnEnergy.com (SOE)** — second largest
- **ChooseEnergy.com (Choose TX)**
- **TexasElectricRates.com (TXER)** — smallest

## Full Funnel

Two parallel paths from session to order:

**Cart path:** Session → ZIP Entry (ZLUR) → Cart Entry (Cart RR) → SSN/Credit Check (SSN Submit Rate) → Order (Cart Conversion)
**Phone path:** Session → Site Queue Call (Site RR) → Phone Order (Site Conversion)

- VC = (Cart Orders + Phone Orders) / Sessions (excludes SERP orders)
- Revenue = Phone Revenue + Cart Revenue
- GCV/Order = average commission per order

## KPI Definitions

All KPIs are session-level rates: SUM(numerator) / SUM(denominator).

- **ZLUR** (ZIP Lookup Rate): zip_entry / session — intent proxy, measures product engagement
- **Cart RR** (Cart Reach Rate): has_cart / session — measures cart reach
- **SSN Submit Rate**: cart_ssn_done / has_cart — pre-credit friction
- **Conversion After Credit**: cart_order / cart_ssn_done — post-credit conversion
- **Cart Conversion**: cart_order / has_cart — end-to-end cart conversion (= SSN Submit Rate × Conversion After Credit)
- **Cart VC**: cart_order / session — cart path visit conversion
- **Phone RR**: queue_call / session — phone reach rate
- **Phone VC**: phone_order / session — phone path visit conversion
- **VC**: total_order / session — total visit conversion (cart + phone)

## Revenue Waterfall (Sequential Substitution)

Phone Revenue = Sessions × Site RR × Site Conversion × Phone GCV/Order
Cart Revenue  = Sessions × Cart RR × Cart Conversion × Cart GCV/Order
Total Revenue = Phone Revenue + Cart Revenue

To attribute why revenue changed, each factor is swapped from Plan to Actual one at a time (Sessions first, then Site RR, etc.). The 7 factor impacts sum exactly to the total revenue gap.

## Personalization Initiatives

- **LP JO** (Landing Page Journey Optimization): ML-based personalization of landing page content. Model vs Holdout experimental design. Agent ID 3577.
- **Grid JO** (Grid Journey Optimization): ML-based personalization of plan grid ordering. Model vs Holdout design. Agent IDs 3378/3401.
- **FMP** (Find My Plan): Guided plan-matching flow via LP agent 3577 classified as "intuitive_explore". Shares the LP holdout as its control. FMP sessions may overlap with Grid JO ("FMP + Grid JO").

The "scaled impact" of an initiative = (model rate − holdout rate) × (initiative traffic share). This measures how much the initiative contributes to the all-in KPI.

## Seasonality

Texas electricity market is highly seasonal:
- **Summer (Jun–Aug):** Peak demand — highest sessions and orders (heat drives plan shopping)
- **Winter (Dec–Feb):** Secondary peak (heating costs)
- **Spring (Mar–May) and Fall (Sep–Nov):** Shoulder seasons, lower demand
- **Weekly pattern:** Mon–Tue highest, weekend lowest

## Channel Characteristics

- **Organic/SEO:** Largest volume channel. Session count driven by Google rankings, impressions, and CTR. Changes are slow (ranking improvements take weeks). Sensitive to algorithm updates and SERP feature changes (AI Overviews).
- **Paid Search:** Second largest. Controllable via bids and budgets. CPC-sensitive. Includes both traditional search and pMax (Google's automated campaign type with black-box auction dynamics).
- **Direct:** Brand recognition traffic. Tends to convert well. Not directly controllable.
- **pMax:** Google Performance Max — automated campaign type. Black box auction dynamics.

## Analytical Concepts

- **Mix effect:** Segment's share of traffic changed (e.g., more Paid Search sessions in the denominator)
- **Rate effect:** Segment's conversion rate changed (e.g., Paid Search converted better this period)
- **"% of total change":** How much of the overall KPI movement a single driver explains
- **Sequential substitution waterfall:** Swap Plan → Actual one factor at a time to attribute dollar impact
- **Pacing:** Projected full-month value from MTD actuals using pacing weights (day-of-month × seasonality)
- **P4WA:** Past 4-period weighted average, used to smooth noise and distinguish signal from random variation"""

_SYSTEM_PROMPT = f"""You are a senior energy marketplace analyst writing a concise performance summary for a weekly business review.

{_BUSINESS_CONTEXT}

## Rules for Your Summary

1. Write 2-4 concise paragraphs suitable for a business audience
2. Lead with the headline using % change language: e.g., "VC improved 3.2% WoW" or "VC declined 1.5% WoW". Do NOT lead with the absolute prior/current rates; speak in terms of the % improvement or decline.
3. When discussing drivers, frame them in terms of how much of the total change they explain. For example: "Paid Search accounted for ~40% of the improvement, driven primarily by a rate lift." or "Direct traffic explained roughly a third of the decline due to a mix shift."
4. Do NOT enumerate prior and current absolute rates for each driver unless specifically asked. Focus on the % of improvement/decline each driver explains and whether it was mix or rate driven.
5. For initiatives, focus on the "scaled impact" — this is the model vs holdout lift multiplied by the initiative's traffic share, representing how much the initiative actually moves the all-in KPI. Compare current vs prior period to say whether initiatives helped more or less this week.
6. Tie drivers to business causality. Instead of "Organic sessions declined", explain "Organic sessions declined because GSC impressions softened — likely seasonal demand softening between the winter billing peak and summer heat."
7. Note seasonality when relevant: spring is a shoulder season, Mon–Tue are peak days, summer drives the highest shopping volume.
8. Use plain business language — avoid jargon like "Shapley" or "counterfactual"
9. Be specific with numbers but keep it readable
10. End with a brief "what to watch" if there's an obvious follow-up question"""


def generate_llm_narrative(
    summary: dict,
    top_drivers: pd.DataFrame,
    initiative_df: pd.DataFrame | None = None,
    initiative_impact: pd.DataFrame | None = None,
    initiative_impact_prior: pd.DataFrame | None = None,
) -> str:
    """Generate a narrative using OpenAI GPT."""
    data_context = _build_data_context(
        summary, top_drivers, initiative_df,
        initiative_impact=initiative_impact,
        initiative_impact_prior=initiative_impact_prior,
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Write a weekly performance summary based on this data. "
                        "Be concise and actionable.\n\n"
                        f"{data_context}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_narrative(summary, top_drivers, initiative_df, error=str(e))


# ═════════════════════════════════════════════════════════════════════════
# Organic Deep-Dive TL;DR — MTD (vs Plan) and WoW (diagnostic flowchart)
# ═════════════════════════════════════════════════════════════════════════
#
# Two separate LLM renderers live here because the MTD and WoW modes have
# fundamentally different framings: MTD leads with the revenue waterfall
# ("why are we off vs Plan?") and only drills into sessions if sessions
# are themselves a driver; WoW leads with the 4-metric flowchart walk
# ("what drove session volatility?"). Both share the rule that the LLM is
# RENDERING a pre-computed structured payload — it never re-diagnoses.


_MTD_TLDR_SYSTEM_PROMPT = """You are a senior SEO analyst writing an MTD (month-to-date) TL;DR paragraph for executives.

CONTEXT
Online energy marketplace serving Texas. Organic is the largest channel. Revenue flows through two funnels:
  • Phone: Sessions × Site RR × Site Conversion × Phone GCV/Order
  • Cart:  Sessions × Cart RR × Cart Conversion × Cart GCV/Order

The payload gives you the rule-based revenue waterfall (Plan → Pacing, with ranked drivers), a 4-metric GSC table (Impressions, Clicks, Sessions, weighted avg rank), and — when Sessions is a top driver — a diagnostic flowchart walk that terminates at a named action-box.

YOUR JOB
Write ONE cohesive narrative paragraph (no bullets, no headings). 4–6 sentences. The paragraph must weave together:

1. **The headline $ story** — what's the vs-Plan gap in dollars and %? Short / ahead, by how much?
2. **The top 1–2 drivers** — which funnel stages opened the gap, quantified in $. If Sessions is a top driver, proceed to step 3; otherwise end with one sentence naming the next check for the biggest non-Sessions driver.
3. **The Sessions story in terms of the four GSC levers** — if Sessions is a driver, explain it as a single arc covering Impressions → Clicks → CTR → Rank. Tell the reader which lever moved, which held, and what that means. Don't say "page concentration is broad" as a cause — that's a routing signal, never the answer. The root cause should be one of: rank loss (and whether scoped to a theme or broad), demand/impression shift, SERP CTR compression, or macro / measurement.
   **When the payload includes `TOP KEYWORD MOVERS`, you MUST name the top 1–2 offending keywords (verbatim, in quotes) and the page-type they land on, inside the ranking sentence.** This is the signal execs care about — generic phrasing like "rankings declined for some queries" is not acceptable when specific high-click keywords are flagged. Example: "the drop is concentrated in the TierCityGEO template, where \"houston electricity rates\" fell from rank 3.2 → 7.8 while \"dallas electricity\" dropped off the first page." Do NOT over-quote — if only one keyword appears in the movers list, name just that one.
4. **The final clause** — one specific next check tied to the story you just told. If keyword movers exist, the next check should reference the named keyword/page-type pair, not a generic "review rankings".

BUSINESS-IMPACT WEIGHTING (critical)
The keyword movers are pre-ranked by business impact (rank delta × click volume × session share). Each mover includes a `session_share` field showing what % of total Organic sessions land on that page type. When describing ranking changes:
- ALWAYS lead with movers on HIGH-session-share page types (≥15% of sessions). These are what actually move the business.
- NEVER lead the summary with a keyword on a LOW-session-share page type (<5% of sessions, e.g. Business, Cart, Spanish) unless it is the ONLY mover. A rank drop on "business electricity rates" (Business page type, ~2% of sessions) is not an impactful finding even if the rank delta is large.
- If ranking losses are spread across page types, say so — but name the page types that matter most by session volume (e.g. CityGEO at 40% of sessions) rather than listing small ones.
- Do NOT describe ranking drops as "widespread" if the impacted page types collectively carry <10% of sessions. That is a low-impact observation, not a headline.

STYLE
- Third-person, factual, zero hedging. No "it appears that", no "there seems to be".
- Every number quoted must come from the payload — do NOT invent figures.
- Bold only the 2–3 numbers that matter most (\\$gap, biggest driver's $ impact, the single GSC lever that broke). Everything else is plain text.
- Escape every dollar sign as `\\$` so Streamlit markdown doesn't treat it as KaTeX.
- Do NOT use em-dash lists or semicolon cascades to disguise a bullet list — it must read as a paragraph.
- No opening phrase like "Here is the TL;DR" or closing phrase like "In summary". Start with the dollar gap sentence.
- If the flowchart is NOT triggered (sessions stable), the paragraph focuses on the Cart/Phone driver(s) and ends with a next check for that driver."""


_WOW_TLDR_SYSTEM_PROMPT = """You are a senior SEO analyst writing a week-over-week (WoW) TL;DR paragraph for executives.

CONTEXT
Online energy marketplace serving Texas. Organic sessions depend on GSC rankings, impressions, and CTR. The payload gives you:
  • Current 7 days ending at the GSC-aligned date, the prior 7 days, and P4WA (prior-4-week average of the same weekday pattern).
  • A 4-metric table: Impressions, Clicks, Sessions, weighted avg rank — with deltas vs prior week and vs P4WA.
  • A diagnostic flowchart walk that traversed the manager's framework (concentration → ranking → impressions → CTR) and landed at a terminal action-box.
  • Click decomposition (impression_effect, ctr_effect, interaction) — use this to say whether the click move was volume-driven or CTR-driven.

YOUR JOB
Write ONE cohesive narrative paragraph (no bullets, no headings). 4–6 sentences. The paragraph must weave together:

1. **The sessions headline** — sessions are ±X% WoW (±Y% vs P4WA). Directional: is the portfolio volatile or steady vs the P4WA baseline?
2. **The four GSC levers, in order** — Impressions, Clicks, CTR, Rank. Walk through them AS ONE STORY: which lever moved and by how much? Which lever held? Was the click move volume-driven (impressions) or efficiency-driven (CTR)? Use the decomposition numbers.
   **When the payload includes `TOP KEYWORD MOVERS`, you MUST name the top 1–2 offending keywords (verbatim, in quotes) and their page-type inside the Rank sentence.** This is the signal execs care about — do not stop at "rank declined." Example: "rank slipped on TierCityGEO where \"houston electricity rates\" fell from 3.2 → 7.8 and \"dallas electricity\" dropped off page 1 entirely." If only one keyword is flagged, cite just that one.
3. **The flowchart verdict as the root cause** — ONE plain-English sentence stating WHY, tied to the flowchart terminal (e.g. "which points to SERP click-loss — AI Overviews or ads compressing top-of-funnel CTR").
4. **The final clause** — one specific next check tied to the terminal. If keyword movers were cited, the next check should reference the named keyword/page-type pair.

BUSINESS-IMPACT WEIGHTING (critical)
The keyword movers are pre-ranked by business impact (rank delta × click volume × session share). Each mover includes a `session_share` field showing what % of total Organic sessions land on that page type. When describing ranking changes:
- ALWAYS lead with movers on HIGH-session-share page types (≥15% of sessions). These are what actually move the business.
- NEVER lead the summary with a keyword on a LOW-session-share page type (<5% of sessions, e.g. Business, Cart, Spanish) unless it is the ONLY mover. A rank drop on "business electricity rates" (Business page type, ~2% of sessions) is not an impactful finding even if the rank delta is large.
- If ranking losses are spread across page types, say so — but name the page types that matter most by session volume (e.g. CityGEO at 40% of sessions) rather than listing small ones.
- Do NOT describe ranking drops as "widespread" if the impacted page types collectively carry <10% of sessions. That is a low-impact observation, not a headline.

STYLE
- Third-person, factual, zero hedging.
- Quote only payload numbers — never invent.
- Bold the 2–3 numbers that matter most. Plain text elsewhere.
- Escape dollar signs as `\\$`.
- One real paragraph. No bullet list disguised with semicolons.
- If the flowchart is NOT triggered (move within stability band), the paragraph is shorter: one sentence on the ±% WoW move, one sentence naming which of the 4 levers was steadiest, and a closing "no action required this week; re-check next Monday" type clause."""


_CUSTOM_TLDR_SYSTEM_PROMPT = """You are a senior SEO analyst writing a custom-window TL;DR paragraph for executives.

The user selected arbitrary date windows — there is no Plan comparison and no P4WA baseline. You have:
  • A 4-metric table (Impressions, Clicks, Sessions, Rank) with current vs prior deltas.
  • Click decomposition (impression_effect, ctr_effect, interaction).
  • A diagnostic flowchart walk with a terminal action-box.

Write ONE cohesive narrative paragraph (no bullets, no headings), 4–6 sentences, weaving the four GSC levers (Impressions → Clicks → CTR → Rank) into one story and closing with the flowchart root-cause verdict plus one next check.

**When the payload includes `TOP KEYWORD MOVERS`, you MUST name the top 1–2 offending keywords (verbatim, in quotes) and their page-type inside the Rank sentence** — generic "rank declined" framing is not acceptable when specific high-click keywords are flagged. Example: "rank slipped on Homepage where \"compare electricity rates texas\" fell from 2.1 → 5.4."

BUSINESS-IMPACT WEIGHTING (critical)
The keyword movers are pre-ranked by business impact (rank delta × click volume × session share). Each mover includes a `session_share` field showing what % of total Organic sessions land on that page type. ALWAYS lead with movers on high-session-share page types (≥15%). NEVER lead with a keyword on a page type carrying <5% of sessions unless it is the only mover. Do NOT describe ranking drops as "widespread" if the impacted page types collectively carry <10% of sessions.

Escape dollar signs as `\\$`. Bold only the 2–3 most important numbers. Start with the sessions or clicks direction — do NOT use preamble."""


def _format_delta(curr: float | None, prior: float | None,
                  *, is_rate: bool = False, is_rank: bool = False) -> str:
    """Format a curr/prior delta as `(abs_diff, pct_diff)` for LLM context."""
    if curr is None or prior is None:
        return "N/A"
    diff = curr - prior
    if is_rank:
        # Rank: lower is better, so invert the emoji interpretation but keep
        # sign as-is. Use 2dp.
        return f"{diff:+.2f} positions"
    if is_rate:
        return f"{diff * 100:+.2f}pp"
    if prior == 0:
        return f"{diff:+,.0f} (Δ%=N/A)"
    return f"{diff:+,.0f} ({diff / prior * 100:+.1f}%)"


def _dump_window(w: dict) -> list[str]:
    if not w:
        return []
    lines = [
        f"Window: {w.get('window_label', 'N/A')} (mode: {w.get('mode', 'N/A')})",
        f"  current: {w.get('curr_start')} → {w.get('curr_end')} ({w.get('curr_days')}d)",
        f"  prior:   {w.get('prior_start')} → {w.get('prior_end')} ({w.get('prior_days')}d)",
    ]
    if w.get("gsc_max_date"):
        lines.append(f"  GSC last fully-reported day: {w['gsc_max_date']}")
    if w.get("truncation_note"):
        lines.append(f"  Truncation: {w['truncation_note']}")
    return lines


def _dump_diagnostic(dx: dict) -> list[str]:
    if not dx:
        return []
    lines = ["\n== FLOWCHART DIAGNOSTIC (authoritative verdict) =="]
    lines.append(f"triggered: {dx.get('triggered')}")
    sd_pct = dx.get("session_delta_pct")
    if sd_pct is not None:
        lines.append(
            f"session_delta: {sd_pct * 100:+.1f}% "
            f"({dx.get('session_delta_abs'):+,.0f} sessions)"
        )
    lines.append(f"trigger_note: {dx.get('trigger_note')}")
    lines.append(f"concentration: {dx.get('concentration')} — {dx.get('concentration_evidence')}")
    lines.append(f"ranking: {dx.get('ranking')} — {dx.get('ranking_evidence')}")
    if dx.get("isolation") and dx.get("isolation") != "n/a":
        lines.append(f"isolation: {dx.get('isolation')} — {dx.get('isolation_evidence')}")
    if dx.get("impressions") and dx.get("impressions") != "unknown":
        lines.append(f"impressions: {dx.get('impressions')} — {dx.get('impressions_evidence')}")
    if dx.get("ctr") and dx.get("ctr") != "unknown":
        lines.append(f"ctr: {dx.get('ctr')} — {dx.get('ctr_evidence')}")
    lines.append(f"terminal_node: {dx.get('terminal_node')}")
    lines.append(f"headline: {dx.get('headline')}")
    lines.append(f"recommended_next_check: {dx.get('recommended_next_check')}")
    return lines


def _dump_top_keyword_movers(movers: list[dict] | None) -> list[str]:
    """Serialize the keyword-rank-tracker movers into LLM context.

    The tracker returns up to 5 keywords whose impression-weighted rank
    moved by ≥ 2 positions on a page-type we can take action on, ordered
    by a severity score (|Δ rank| × log(prior_clicks+1) × session_share).
    Session share weights ensure keywords on high-traffic page types
    (CityGEO, Homepage) rank above keywords on low-traffic page types
    (Business, Cart) in the narrative. Emitting this into the prompt lets
    the narrative name specific keywords and page-types in the Sessions /
    ranking story.
    """
    if not movers:
        return []
    lines = [
        "\n== TOP KEYWORD MOVERS (|Δ rank| ≥ 2 positions, ranked by "
        "severity = |Δrank| × log(prior_clicks+1) × session_share) ==",
        "These are ALREADY ranked by business impact (rank move × click "
        "volume × session share). Call out the top 1–2 by severity. "
        "session_share tells you what % of Organic sessions land on that "
        "page type — use it to contextualize importance. A rank drop on a "
        "page type with <5% of sessions is low-impact and should NOT lead "
        "the summary. \"dropped\" means the keyword fell out of the current "
        "window entirely.",
    ]
    for m in movers:
        pieces = [
            f"  query=\"{m.get('query', '?')}\"",
            f"page_type={m.get('landing_page_type', '?')}",
        ]
        if m.get("session_share_pct"):
            pieces.append(f"session_share={m['session_share_pct']}")
        if m.get("prior_rank") is not None:
            pieces.append(f"prior_rank={m['prior_rank']:.1f}")
        if m.get("dropped_out"):
            pieces.append("curr_rank=DROPPED")
        elif m.get("curr_rank") is not None:
            pieces.append(f"curr_rank={m['curr_rank']:.1f}")
        if m.get("rank_delta") is not None:
            pieces.append(f"Δrank={m['rank_delta']:+.1f}")
        if m.get("prior_clicks") is not None:
            pieces.append(f"prior_clicks={int(m['prior_clicks']):,}")
        if m.get("curr_clicks") is not None:
            pieces.append(f"curr_clicks={int(m['curr_clicks']):,}")
        if m.get("click_delta_pct"):
            pieces.append(f"Δclicks%={m['click_delta_pct']}")
        lines.append(" · ".join(pieces))
    return lines


def _dump_4metric_table(m: dict) -> list[str]:
    """Render the 4-metric table (impr/clicks/sessions/rank) for the LLM."""
    if not m:
        return []
    out = ["\n== 4-METRIC TABLE =="]
    for key, label in (
        ("impressions", "Impressions"),
        ("clicks",      "Clicks"),
        ("sessions",    "Sessions"),
        ("rank",        "Weighted avg rank"),
    ):
        row = m.get(key) or {}
        if not row:
            continue
        bits = [f"{label}:"]
        if row.get("curr") is not None:
            bits.append(f"curr={row['curr']}")
        if row.get("prior") is not None:
            bits.append(f"prior={row['prior']}")
        if row.get("delta") is not None:
            bits.append(f"Δ={row['delta']}")
        if row.get("pct_change") is not None:
            bits.append(f"Δ%={row['pct_change']}")
        if row.get("p4wa") is not None:
            bits.append(f"p4wa={row['p4wa']}")
        if row.get("pct_change_vs_p4wa") is not None:
            bits.append(f"Δ% vs P4WA={row['pct_change_vs_p4wa']}")
        out.append("  " + " · ".join(bits))
    return out


def _format_keyword_movers_sentence(movers: list[dict] | None) -> str:
    """Render a single exec-facing sentence naming the top 1–2 keyword
    movers. Used by both rule-based fallbacks when the LLM is unavailable
    — the point is to guarantee keyword-specific language in the TL;DR
    paragraph even in the degraded path.

    Returns an empty string when no movers exist so callers can filter it
    out without extra branching.
    """
    if not movers:
        return ""
    top = movers[:2]
    fragments: list[str] = []
    for m in top:
        q = m.get("query", "")
        pt = m.get("landing_page_type", "")
        if m.get("dropped_out"):
            fragments.append(
                f'"{q}" fell off the first page on {pt}'
            )
        else:
            prior = m.get("prior_rank")
            curr = m.get("curr_rank")
            delta = m.get("rank_delta")
            if prior is not None and curr is not None:
                fragments.append(
                    f'"{q}" on {pt} slipped from rank {prior:.1f} → {curr:.1f}'
                )
            elif delta is not None:
                fragments.append(
                    f'"{q}" on {pt} moved {delta:+.1f} positions'
                )
            else:
                fragments.append(f'"{q}" on {pt}')
    return (
        "The rank loss is concentrated in top-click keywords — "
        + " and ".join(fragments)
        + "."
    )


# ── MTD builder ──────────────────────────────────────────────────────────


def _build_mtd_tldr_context(payload: dict) -> str:
    lines: list[str] = []
    lines.extend(_dump_window(payload.get("window", {})))

    # Revenue waterfall drivers (primary content for MTD)
    wf = payload.get("waterfall") or {}
    if wf:
        lines.append("\n== REVENUE WATERFALL (Plan → Pacing) ==")
        lines.append(f"plan_revenue: ${wf.get('plan_revenue', 0):,.0f}")
        lines.append(f"pacing_revenue: ${wf.get('pacing_revenue', 0):,.0f}")
        lines.append(f"total_gap: ${wf.get('total_gap', 0):,.0f}")
        lines.append(f"pct_of_plan: {wf.get('pct_of_plan', 0) * 100:+.1f}%")
        lines.append("Drivers ranked by absolute \\$ impact (largest first):")
        for d in wf.get("ranked_drivers", []):
            lines.append(
                f"  {d['rank']}. {d['name']}: {d['impact_formatted']} "
                f"({d['pct_of_gap']} of gap)"
                + (f" · Plan={d['plan_formatted']}, Pacing={d['pacing_formatted']}, "
                   f"vs Plan={d['vs_plan_pct']}" if d.get('plan_formatted') else "")
            )

    # Flowchart diagnostic (only when sessions is a driver)
    lines.extend(_dump_diagnostic(payload.get("diagnostic") or {}))

    # 4-metric GSC table (useful supporting context even when not in headline)
    lines.extend(_dump_4metric_table(payload.get("four_metric") or {}))

    # Page-type movers (only when concentration fired)
    pt = payload.get("page_type_movers", [])
    if pt:
        lines.append("\n== Top page-type click movers ==")
        lines.append("  (session_share = % of Organic sessions on this page type)")
        for r in pt:
            sess_share = r.get("session_share_pct", "")
            share_str = f" · session_share={sess_share}" if sess_share else ""
            lines.append(
                f"  {r['landing_page_type']}: {r['click_delta']:+,.0f} clicks "
                f"({r['click_delta_pct']}){share_str}"
            )

    # Keyword-level rank movers — lets the LLM name specific keywords.
    lines.extend(_dump_top_keyword_movers(payload.get("top_keyword_movers")))

    return "\n".join(lines)


def generate_mtd_vs_plan_tldr(payload: dict) -> str:
    """MTD TL;DR: leads with the revenue waterfall's top drivers. Drills
    into Sessions only when Sessions is a top driver.
    """
    context = _build_mtd_tldr_context(payload)
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _MTD_TLDR_SYSTEM_PROMPT},
                {"role": "user", "content":
                    "Synthesize the MTD TL;DR as ONE narrative paragraph "
                    "(no bullets, no headings). Weave together the vs-Plan "
                    "gap, the top 1–2 drivers, the Sessions story (as "
                    "Impressions → Clicks → CTR → Rank when Sessions is a "
                    "driver), and a final next check. 4–6 sentences.\n\n"
                    + context},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_mtd_tldr(payload, error=str(e))


def _fallback_mtd_tldr(payload: dict, error: str = "") -> str:
    """Rule-based fallback — renders as a single narrative paragraph
    so the shape matches what the LLM would produce.
    """
    wf = payload.get("waterfall") or {}
    dx = payload.get("diagnostic") or {}
    m = payload.get("four_metric") or {}

    if not wf:
        msg = "*Waterfall data unavailable — cannot render MTD TL;DR.*"
        if error:
            msg += f" *(LLM error: {error[:80]}.)*"
        return msg

    gap = wf.get("total_gap", 0)
    plan = wf.get("plan_revenue", 0)
    pacing = wf.get("pacing_revenue", 0)
    pct = wf.get("pct_of_plan", 0) * 100
    direction = "short" if gap < 0 else "ahead"
    drivers = wf.get("ranked_drivers", [])
    top = drivers[0] if drivers else None
    second = drivers[1] if len(drivers) > 1 else None

    parts: list[str] = []
    parts.append(
        f"Pacing is **\\${pacing:,.0f}** vs Plan **\\${plan:,.0f}**, "
        f"**{direction} by \\${abs(gap):,.0f} ({pct:+.1f}%)**."
    )

    if top:
        parts.append(
            f"The gap is led by **{top['name']}** "
            f"({top['impact_formatted']}, {top['pct_of_gap']} of the gap)"
            + (f", with **{second['name']}** close behind "
               f"({second['impact_formatted']})." if second else ".")
        )

    # Sessions story — only when Sessions is a driver AND the flowchart fired.
    sessions_is_driver = top and top["name"] == "Sessions"
    if sessions_is_driver and dx.get("triggered"):
        impr = (m.get("impressions") or {}).get("pct_change")
        clicks = (m.get("clicks") or {}).get("pct_change")
        rank_delta = (m.get("rank") or {}).get("delta")
        sess_pct = (m.get("sessions") or {}).get("pct_change")
        lever_bits: list[str] = []
        if impr:
            lever_bits.append(f"impressions {impr}")
        if clicks:
            lever_bits.append(f"clicks {clicks}")
        if rank_delta:
            lever_bits.append(f"rank {rank_delta}")
        if sess_pct:
            lever_bits.append(f"sessions {sess_pct}")
        if lever_bits:
            parts.append("On the GSC side, " + ", ".join(lever_bits) + ".")

        # Name specific keyword movers so execs see the real signal.
        kw_sentence = _format_keyword_movers_sentence(
            payload.get("top_keyword_movers")
        )
        if kw_sentence:
            parts.append(kw_sentence)

        headline = dx.get("headline", "").strip()
        if headline:
            parts.append(headline.rstrip(".") + ".")

    next_check = (
        dx.get("recommended_next_check")
        if sessions_is_driver and dx.get("recommended_next_check")
        else f"Drill into {top['name']}" if top else "Drill into the top driver"
    )
    parts.append(f"Next check: {next_check.strip().rstrip('.')}.")

    paragraph = " ".join(p.rstrip() for p in parts if p)
    if error:
        paragraph += f"\n\n*(LLM TL;DR unavailable — fell back to rule-based synthesis.)*"
    return paragraph


# ── WoW / Custom builder ──────────────────────────────────────────────────


def _build_wow_tldr_context(payload: dict) -> str:
    lines: list[str] = []
    lines.extend(_dump_window(payload.get("window", {})))
    lines.extend(_dump_4metric_table(payload.get("four_metric") or {}))
    lines.extend(_dump_diagnostic(payload.get("diagnostic") or {}))

    # Click decomposition (explains Clicks bullet)
    decomp = payload.get("click_decomp") or {}
    if decomp:
        lines.append("\n== CLICK DECOMPOSITION (prior → current) ==")
        lines.append(
            f"  impression_effect: {decomp.get('impression_effect', 0):+,.0f} clicks"
        )
        lines.append(
            f"  ctr_effect:        {decomp.get('ctr_effect', 0):+,.0f} clicks"
        )
        lines.append(
            f"  interaction:        {decomp.get('interaction', 0):+,.0f} clicks"
        )

    pt = payload.get("page_type_movers", [])
    if pt:
        lines.append("\n== Top page-type click movers ==")
        for r in pt:
            lines.append(
                f"  {r['landing_page_type']}: {r['click_delta']:+,.0f} clicks "
                f"({r['click_delta_pct']})"
            )

    # Keyword-level rank movers — lets the LLM name specific keywords.
    lines.extend(_dump_top_keyword_movers(payload.get("top_keyword_movers")))
    return "\n".join(lines)


def generate_wow_tldr(payload: dict) -> str:
    """WoW TL;DR: ONE paragraph weaving Impressions → Clicks → CTR → Rank
    into a single narrative that ends with the flowchart verdict.
    """
    context = _build_wow_tldr_context(payload)
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _WOW_TLDR_SYSTEM_PROMPT},
                {"role": "user", "content":
                    "Synthesize the WoW TL;DR as ONE narrative paragraph (no "
                    "bullets, no headings). Weave the four GSC levers "
                    "(Impressions → Clicks → CTR → Rank) into one story, "
                    "tie it to the flowchart terminal, and close with a "
                    "next check. 4–6 sentences.\n\n" + context},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_wow_tldr(payload, error=str(e))


def generate_custom_tldr(payload: dict) -> str:
    """Custom-window TL;DR: same paragraph format as WoW minus the P4WA column."""
    context = _build_wow_tldr_context(payload)
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _CUSTOM_TLDR_SYSTEM_PROMPT},
                {"role": "user", "content":
                    "Synthesize the custom-window TL;DR as ONE narrative "
                    "paragraph (no bullets, no headings). 4–6 sentences.\n\n"
                    + context},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_wow_tldr(payload, error=str(e))


def _fallback_wow_tldr(payload: dict, error: str = "") -> str:
    """Rule-based fallback — single narrative paragraph to match the
    LLM's target shape.
    """
    dx = payload.get("diagnostic") or {}
    m = payload.get("four_metric") or {}
    decomp = payload.get("click_decomp") or {}

    sess = m.get("sessions") or {}
    impr = m.get("impressions") or {}
    clicks = m.get("clicks") or {}
    rank = m.get("rank") or {}

    sess_pct = sess.get("pct_change")
    sess_p4wa = sess.get("pct_change_vs_p4wa")
    impr_pct = impr.get("pct_change")
    impr_p4wa = impr.get("pct_change_vs_p4wa")
    clicks_pct = clicks.get("pct_change")
    rank_delta = rank.get("delta")

    parts: list[str] = []

    # 1. Sessions headline.
    if sess_pct is not None and sess_p4wa is not None:
        parts.append(f"Sessions are **{sess_pct} WoW** ({sess_p4wa} vs P4WA).")
    elif sess_pct is not None:
        parts.append(f"Sessions are **{sess_pct} WoW**.")

    # 2. The four GSC levers, woven.
    lever_parts: list[str] = []
    if impr_pct is not None:
        lever_parts.append(f"impressions **{impr_pct}**" +
                           (f" ({impr_p4wa} vs P4WA)" if impr_p4wa else ""))
    if clicks_pct is not None:
        lever_parts.append(f"clicks **{clicks_pct}**")
    if rank_delta is not None:
        lever_parts.append(f"rank moved {rank_delta}")
    if lever_parts:
        parts.append("On the GSC side, " + ", ".join(lever_parts) + ".")

    # 3. Impressions vs CTR attribution.
    impression_effect = decomp.get("impression_effect")
    ctr_effect = decomp.get("ctr_effect")
    if impression_effect is not None and ctr_effect is not None:
        if abs(impression_effect) > abs(ctr_effect) * 1.5:
            parts.append("The click move is volume-driven (impressions).")
        elif abs(ctr_effect) > abs(impression_effect) * 1.5:
            parts.append("The click move is efficiency-driven (CTR).")

    # 3b. Name specific keyword movers so execs see the real signal.
    kw_sentence = _format_keyword_movers_sentence(
        payload.get("top_keyword_movers")
    )
    if kw_sentence:
        parts.append(kw_sentence)

    # 4. Flowchart verdict + next check.
    headline = (dx.get("headline") or "").strip()
    if headline:
        parts.append(headline.rstrip(".") + ".")
    next_check = (dx.get("recommended_next_check") or "").strip()
    if next_check:
        parts.append(f"Next check: {next_check.rstrip('.')}.")

    paragraph = " ".join(p.rstrip() for p in parts if p) or "TL;DR unavailable."
    if error:
        paragraph += "\n\n*(LLM TL;DR unavailable — fell back to rule-based synthesis.)*"
    return paragraph


def build_chat_system_prompt(
    summary: dict,
    top_drivers: pd.DataFrame,
    initiative_df: pd.DataFrame | None = None,
    funnel_table_text: str = "",
    initiative_impact: pd.DataFrame | None = None,
    initiative_impact_prior: pd.DataFrame | None = None,
) -> str:
    """
    Build a rich system prompt for the chat agent that includes all
    the current analysis context so the user can ask follow-up questions.
    """
    data_context = _build_data_context(
        summary, top_drivers, initiative_df,
        initiative_impact=initiative_impact,
        initiative_impact_prior=initiative_impact_prior,
    )

    return f"""{_SYSTEM_PROMPT}

You are now in a conversational mode. The user has just reviewed the analysis below and wants to ask follow-up questions. Answer based on the data provided. If you don't have enough information to answer, say so clearly and suggest what additional data or analysis would help.

Be conversational but precise. Speak in terms of % change (e.g., "VC improved 3%") rather than absolute rates. When discussing drivers, frame them as "X explained ~Y% of the improvement." Reference specific numbers from the data when relevant.

When the user asks "why" a metric moved, connect the dots through the funnel: sessions drive volume, reach rates drive cart/phone entries, conversion rates drive orders. Always tie back to revenue impact when possible.

--- CURRENT ANALYSIS CONTEXT ---

{data_context}

{f"Full funnel summary:{chr(10)}{funnel_table_text}" if funnel_table_text else ""}

--- END CONTEXT ---"""


def stream_chat_response(
    messages: list[dict],
    system_prompt: str,
):
    """
    Stream a chat response from OpenAI. Yields text chunks.
    messages should be the conversation history (role: user/assistant).
    """
    client = _get_client()
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=full_messages,
        temperature=0.4,
        max_tokens=1000,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def run_analyst_chat(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict],
    tool_executor,
    max_rounds: int = 3,
) -> tuple[str, list[dict]]:
    """
    Chat completion with an OpenAI tool-calling loop.

    Returns ``(final_text, tool_outputs)`` where *tool_outputs* is a list of
    ``{"tool": str, "explanation": str, "result_str": str, "result_obj": Any}``.
    Intermediate tool calls are resolved automatically (up to *max_rounds*).
    """
    client = _get_client()
    full_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    tool_outputs: list[dict] = []

    response = None
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=full_messages,
            tools=tools,
            temperature=0.3,
            max_tokens=2000,
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            full_messages.append(msg)
            for tc in msg.tool_calls:
                result_str, result_obj, explanation = tool_executor(
                    tc.function.name, tc.function.arguments,
                )
                tool_outputs.append({
                    "tool": tc.function.name,
                    "explanation": explanation,
                    "result_str": result_str,
                    "result_obj": result_obj,
                })
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            return msg.content or "", tool_outputs

    last = response.choices[0].message.content if response else ""
    return (
        last or "Analysis could not be completed within the allowed steps.",
        tool_outputs,
    )


def _fallback_narrative(
    summary: dict,
    top_drivers: pd.DataFrame,
    initiative_df: pd.DataFrame | None = None,
    error: str = "",
) -> str:
    """Structured fallback if OpenAI is unavailable."""
    kpi_def = KPIS[summary["kpi"]]
    pct_ch = _pct_change(summary["delta"], summary["prior_rate"])
    direction = "improved" if summary["delta"] > 0 else "declined" if summary["delta"] < 0 else "was flat"
    parts = []

    if error:
        parts.append(f"*LLM unavailable ({error}) — showing structured summary.*\n")

    parts.append(
        f"**{kpi_def.name}** {direction} **{pct_ch}** WoW "
        f"({_fmt_pp(summary['delta'])})."
    )

    total_delta = summary["delta"]
    if not top_drivers.empty and total_delta != 0:
        positives = top_drivers[top_drivers["total_contribution"] > 0].sort_values(
            "total_contribution", ascending=False
        )
        negatives = top_drivers[top_drivers["total_contribution"] < 0].sort_values(
            "total_contribution"
        )

        if not positives.empty:
            parts.append("\n**Positive drivers:**")
            for _, row in positives.head(5).iterrows():
                dim_name = _dim_label(row["dimension"])
                pct_of = _driver_pct_of_total(row["total_contribution"], total_delta)
                parts.append(f"- **{dim_name} = {row['segment']}**: explains ~{pct_of} of the change")

        if not negatives.empty:
            parts.append("\n**Negative drivers:**")
            for _, row in negatives.head(5).iterrows():
                dim_name = _dim_label(row["dimension"])
                pct_of = _driver_pct_of_total(row["total_contribution"], total_delta)
                parts.append(f"- **{dim_name} = {row['segment']}**: explains ~{pct_of} of the change")

    return "\n".join(parts)
