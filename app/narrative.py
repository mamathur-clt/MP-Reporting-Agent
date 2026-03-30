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


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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


_SYSTEM_PROMPT = """You are a senior energy marketplace analyst writing a concise performance summary for a weekly business review.

Context: You work for an online energy marketplace that helps Texas consumers compare and enroll in electricity plans. The business operates four sites (ChooseTexasPower.org is the largest). The funnel goes: Session → ZIP entry → Cart → SSN/Credit → Order. There's also a parallel phone funnel.

Personalization initiatives running on-site:
- LP JO (Landing Page Journey Optimization): personalizes the landing page via ML model vs holdout
- Grid JO (Grid Journey Optimization): personalizes the plan grid via ML model vs holdout
- FMP (Find My Plan): guided plan-matching flow, classified via the LP agent (agent 3313) as "intuitive_explore". FMP shares the LP holdout as its control group. FMP sessions may also overlap with Grid JO ("FMP + Grid JO").

Key analytical concepts:
- "Mix effect" means the segment's share of traffic changed (e.g., more Paid Search sessions)
- "Rate effect" means the segment's conversion rate changed (e.g., Paid Search converted better)
- "% of total change" tells you how much of the overall KPI movement is explained by a single driver

Rules for your summary:
1. Write 2-4 concise paragraphs suitable for a business audience
2. Lead with the headline using % change language: e.g., "VC improved 3.2% WoW" or "VC declined 1.5% WoW". Do NOT lead with the absolute prior/current rates; speak in terms of the % improvement or decline.
3. When discussing drivers, frame them in terms of how much of the total change they explain. For example: "Paid Search accounted for ~40% of the improvement, driven primarily by a rate lift." or "Direct traffic explained roughly a third of the decline due to a mix shift."
4. Do NOT enumerate prior and current absolute rates for each driver unless specifically asked. Focus on the % of improvement/decline each driver explains and whether it was mix or rate driven.
5. For initiatives, focus on the "scaled impact" — this is the model vs holdout lift multiplied by the initiative's traffic share, representing how much the initiative actually moves the all-in KPI. Compare current vs prior period to say whether initiatives helped more or less this week.
6. Use plain business language — avoid jargon like "Shapley" or "counterfactual"
7. Be specific with numbers but keep it readable
8. End with a brief "what to watch" if there's an obvious follow-up question"""


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
