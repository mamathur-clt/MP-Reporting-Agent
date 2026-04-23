"""
Rule-based SEO session diagnostic walker.

Implements the manager's diagnostic flowchart verbatim:

    SEO sessions declined
        └─ Check whether decline was driven by specific page(s) / domain(s)
            └─ Was there an outsized impact from specific page types / domains?
                ├─ Yes            → Identify theme across impacted pages
                └─ No concentration → Review broader sitewide keyword / page patterns
                        (both branches converge)
                └─ Check keyword-level ranking changes for impacted pages/theme
                    └─ Were there keyword ranking drops?
                        ├─ Yes, for specific keyword categories
                        │       → Optimize for affected keyword categories
                        ├─ Yes, across many keyword types
                        │   └─ Was ranking loss isolated to a few pages or many pages?
                        │       ├─ Few pages → Review whether content is outdated /
                        │       │              less competitive → Refresh and broadly
                        │       │              optimize content
                        │       └─ Many pages → Investigate external forces
                        │                       (algorithm, competitor, SERP)
                        │                       → Document likely external cause and
                        │                         define response plan
                        └─ No notable keyword-level ranking drops
                            └─ Check search interest (GSC impressions on kws that
                               were & remain page-1 / top-5)
                                └─ Did impressions decline between periods?
                                    ├─ Yes → Attribute decline primarily to reduced
                                    │        search interest / macro demand changes
                                    └─ No, impressions stable
                                        └─ Check CTR / click efficiency
                                            └─ Did CTR fall?
                                                ├─ Yes → Investigate SERP click-loss
                                                │        drivers (AI Overviews, paid
                                                │        ads, snippets, SERP features)
                                                └─ No → Review other macro /
                                                         measurement / traffic-mix
                                                         factors

The walker consumes the numbers already computed for the Organic tab
(GSC click decomposition, page-type movers, weighted-rank deltas) and
returns both a `DiagnosticReport` and a Graphviz DOT string that
highlights the traversed path. The LLM narrative layer never re-
diagnoses — it just renders the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


# ---------------------------------------------------------------------------
# Thresholds (documented; tune in one place)
# ---------------------------------------------------------------------------

MATERIAL_SESSION_DELTA_PCT = 0.03
MATERIAL_SESSION_DELTA_ABS = 1_000

# Concentration routing: the manager's chart treats "Yes" as a single
# (or small cluster of) page types being the dominant driver of the click
# delta. We require BOTH:
#   • top-1 page type alone carries ≥ 50% of the |click delta|, AND
#   • top-1 / top-2 ratio ≥ 1.8 (i.e. #1 is materially bigger than #2).
# That second check prevents a fairly flat distribution like
# 42 / 22 / 19 / 17 from being misread as "concentrated".
CONCENTRATION_TOP_N = 2
CONCENTRATION_TOP1_SHARE = 0.50
CONCENTRATION_TOP1_OVER_TOP2 = 1.8

# Ranking-drop gate (hybrid): either impression-weighted rank worsened
# ≥ 0.5 positions OR ≥ 10% of previously page-1 queries fell below page 1.
RANK_THRESHOLD_POSITIONS = 0.5
PAGE1_CHURN_THRESHOLD = 0.10

# "Many keyword types" vs "specific keyword categories" — using page-type
# breadth as the proxy (confirmed with the user). 3+ page types moving in
# the same direction ⇒ broad. 1–2 ⇒ specific.
BROAD_PAGE_TYPE_COUNT = 3

# Ranking loss isolated to "few pages" vs "many pages" — measured on the
# same page-type movers (post-routing this is a scope question).
FEW_PAGES_MAX = 2

# Impression gate.
IMPRESSION_MOVE_THRESHOLD_PCT = 0.05

# CTR gate.
CTR_MOVE_THRESHOLD_PCT = 0.05


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Outcomes for each flowchart diamond.
ConcentrationVerdict = Literal["concentrated", "no_concentration", "unknown"]
RankingVerdict = Literal[
    "specific_categories",    # ranking drops concentrated in 1–2 page types
    "many_types",             # drops across 3+ page types (needs scope sub-gate)
    "stable",                 # no notable ranking drops
    "unknown",                # insufficient data
]
IsolationVerdict = Literal["few_pages", "many_pages", "n/a"]
ImpressionVerdict = Literal["declined", "stable", "grew", "unknown"]
CtrVerdict = Literal["fell", "stable", "rose", "unknown"]

# The six terminal action-box strings from the manager's flowchart. These
# are exactly what gets shown as the headline (verbatim from the chart).
TerminalNode = Literal[
    "optimize_affected_keyword_categories",
    "refresh_and_broadly_optimize_content",
    "document_external_cause_define_response",
    "attribute_to_reduced_search_interest",
    "investigate_serp_click_loss_drivers",
    "review_other_macro_measurement_factors",
    "session_stable",                 # pre-flowchart exit (no material move)
    "insufficient_data",               # can't evaluate any gate
]

_TERMINAL_LABELS: dict[str, str] = {
    "optimize_affected_keyword_categories":
        "Optimize for affected keyword categories",
    "refresh_and_broadly_optimize_content":
        "Refresh and broadly optimize content",
    "document_external_cause_define_response":
        "Document likely external cause & define response plan",
    "attribute_to_reduced_search_interest":
        "Attribute decline to reduced search interest / macro demand",
    "investigate_serp_click_loss_drivers":
        "Investigate SERP click-loss drivers (AI Overviews, ads, snippets)",
    "review_other_macro_measurement_factors":
        "Review other macro / measurement / traffic-mix factors",
    "session_stable": "Sessions stable — no root-cause walk required",
    "insufficient_data": "Insufficient data to evaluate the flowchart",
}


@dataclass
class GateDecision:
    """A single diamond in the flowchart — which outcome did we take?"""

    name: str                                    # human-readable gate label
    verdict: str                                 # literal outcome string
    evidence: str                                # one-sentence observation
    edge_label: str = ""                         # how this connects to next gate


@dataclass
class DiagnosticReport:
    """Structured output of the flowchart walk."""

    # Trigger
    triggered: bool
    session_delta_pct: float | None
    session_delta_abs: float | None
    session_direction: Literal["up", "down", "flat"] = "flat"
    trigger_note: str = ""

    # Verdicts for each diamond (in traversal order)
    concentration: ConcentrationVerdict = "unknown"
    concentration_evidence: str = ""
    concentration_theme: list[str] = field(default_factory=list)

    ranking: RankingVerdict = "unknown"
    ranking_evidence: str = ""

    isolation: IsolationVerdict = "n/a"
    isolation_evidence: str = ""

    impressions: ImpressionVerdict = "unknown"
    impressions_evidence: str = ""

    ctr: CtrVerdict = "unknown"
    ctr_evidence: str = ""

    # Terminal action-box
    terminal_node: TerminalNode = "insufficient_data"
    headline: str = ""                           # one-sentence synthesis for the TL;DR
    recommended_next_check: str = ""

    # Node IDs visited — used to color the graphviz chart.
    visited_nodes: list[str] = field(default_factory=list)
    gate_decisions: list[GateDecision] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render_graphviz(self) -> str:
        """Return a DOT string of the manager's flowchart with the visited
        path highlighted. Used by `st.graphviz_chart`.
        """
        return _build_graphviz_dot(self.visited_nodes)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:+.1f}%"


def _fmt_clicks(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{x:+,.0f}"


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def walk_diagnostic_tree(
    *,
    session_curr: float | None,
    session_prior: float | None,
    impression_effect: float | None,
    ctr_effect: float | None,
    pct_change_impressions: float | None,
    pct_change_ctr: float | None,
    curr_rank: float | None,
    prior_rank: float | None,
    page_type_movers: list[dict] | None,
    page1_churn_pct: float | None,
    window_label: str,
    session_share_map: dict[str, float] | None = None,
) -> DiagnosticReport:
    """
    Walk the flowchart given the numbers from the Organic tab.

    Parameters
    ----------
    session_curr, session_prior : Organic session totals (finance) for the
        two comparison windows.
    impression_effect, ctr_effect : Outputs of `compute_click_decomposition`.
    pct_change_impressions, pct_change_ctr : Deltas in % terms.
    curr_rank, prior_rank : Impression-weighted avg rank.
    page_type_movers : list of dicts from the Section-5 page-type table.
        Must contain `landing_page_type`, `click_delta`, `clicks_curr`,
        `clicks_prior`.
    page1_churn_pct : Share of previously page-1 queries (rank ≤ 4) that
        fell off page 1 (rank > 10) in the current window. Can be None.
    window_label : Human-readable window ("Apr 1–Apr 14 vs Mar 1–Mar 14").
    """

    # ── Trigger ────────────────────────────────────────────────────────
    session_delta_abs: float | None = None
    session_delta_pct: float | None = None
    direction: Literal["up", "down", "flat"] = "flat"
    if session_curr is not None and session_prior is not None and session_prior > 0:
        session_delta_abs = float(session_curr - session_prior)
        session_delta_pct = session_delta_abs / session_prior
        if session_delta_pct > 0:
            direction = "up"
        elif session_delta_pct < 0:
            direction = "down"

    triggered = (
        session_delta_pct is not None
        and (abs(session_delta_pct) >= MATERIAL_SESSION_DELTA_PCT
             or abs(session_delta_abs or 0) >= MATERIAL_SESSION_DELTA_ABS)
    )

    if session_delta_pct is None:
        trigger_note = (
            "Session totals unavailable from finance — diagnostic walk skipped. "
            "Falling back to GSC click signals only."
        )
    elif not triggered:
        trigger_note = (
            f"Sessions changed only {_fmt_pct(session_delta_pct)} over {window_label} "
            f"({session_delta_abs:+,.0f}) — within the ±{MATERIAL_SESSION_DELTA_PCT * 100:.0f}% "
            "stability band."
        )
    else:
        trigger_note = (
            f"Sessions {direction} **{_fmt_pct(session_delta_pct)}** over {window_label} "
            f"({session_delta_abs:+,.0f}) — walking the diagnostic flowchart…"
        )

    if not triggered:
        return DiagnosticReport(
            triggered=False,
            session_delta_pct=session_delta_pct,
            session_delta_abs=session_delta_abs,
            session_direction=direction,
            trigger_note=trigger_note,
            terminal_node="session_stable",
            headline=trigger_note,
            recommended_next_check="",
            visited_nodes=["start", "session_stable"],
        )

    # ── 1. Concentration routing ───────────────────────────────────────
    conc = _evaluate_concentration(page_type_movers)
    gate_decisions = [GateDecision(
        name="Was there an outsized impact from specific page types / domains?",
        verdict=conc["verdict"],
        evidence=conc["evidence"],
        edge_label={"concentrated": "Yes", "no_concentration": "No clear concentration"}
            .get(conc["verdict"], ""),
    )]
    visited_nodes: list[str] = ["start", "initial_check", "diamond_concentration"]
    if conc["verdict"] == "concentrated":
        visited_nodes.append("identify_theme")
    elif conc["verdict"] == "no_concentration":
        visited_nodes.append("review_sitewide")
    visited_nodes.append("check_keyword_rankings")

    # ── 2. Ranking gate ────────────────────────────────────────────────
    rank = _evaluate_ranking(
        curr_rank=curr_rank,
        prior_rank=prior_rank,
        page_type_movers=page_type_movers,
        page1_churn_pct=page1_churn_pct,
        concentration_verdict=conc["verdict"],
        concentration_theme=conc.get("theme", []),
        session_share_map=session_share_map,
    )
    gate_decisions.append(GateDecision(
        name="Were there keyword ranking drops?",
        verdict=rank["verdict"],
        evidence=rank["evidence"],
        edge_label={
            "specific_categories": "Yes, for specific keyword categories",
            "many_types": "Yes, across many keyword types",
            "stable": "No notable keyword-level ranking drops",
            "unknown": "Rank data unavailable",
        }[rank["verdict"]],
    ))
    visited_nodes.append("diamond_rank_drops")

    # --- Ranking branch A: specific categories → keyword-category fix ---
    if rank["verdict"] == "specific_categories":
        visited_nodes.append("terminal_optimize_keyword_categories")
        headline = _build_headline_specific_categories(
            direction=direction,
            session_delta_pct=session_delta_pct,
            ranking_theme=conc.get("theme", []),
            rank_evidence=rank["evidence"],
        )
        return DiagnosticReport(
            triggered=True,
            session_delta_pct=session_delta_pct,
            session_delta_abs=session_delta_abs,
            session_direction=direction,
            trigger_note=trigger_note,
            concentration=conc["verdict"],
            concentration_evidence=conc["evidence"],
            concentration_theme=conc.get("theme", []),
            ranking=rank["verdict"],
            ranking_evidence=rank["evidence"],
            terminal_node="optimize_affected_keyword_categories",
            headline=headline,
            recommended_next_check=(
                "Pull the query-level rank table for the affected page types and "
                "ship refreshed titles / on-page copy targeting the lost terms."
            ),
            visited_nodes=visited_nodes,
            gate_decisions=gate_decisions,
        )

    # --- Ranking branch B: many types → scope sub-gate ---
    if rank["verdict"] == "many_types":
        isolation = _evaluate_isolation(page_type_movers)
        gate_decisions.append(GateDecision(
            name="Was ranking loss isolated to a few pages or many pages?",
            verdict=isolation["verdict"],
            evidence=isolation["evidence"],
            edge_label={"few_pages": "Few pages only", "many_pages": "Many pages"}
                .get(isolation["verdict"], ""),
        ))
        visited_nodes.append("diamond_isolation")
        if isolation["verdict"] == "few_pages":
            visited_nodes.extend(["review_content_outdated",
                                  "terminal_refresh_optimize_content"])
            terminal = "refresh_and_broadly_optimize_content"
            headline = (
                f"Ranking loss is limited to a handful of pages "
                f"({isolation['evidence']}) — likely outdated or less-competitive "
                "content on those specific URLs."
            )
            next_check = (
                "Audit the top-losing URLs for title tags, content freshness, "
                "and intent match. Refresh copy and internal linking."
            )
        else:  # many_pages
            visited_nodes.extend(["investigate_external_forces",
                                  "terminal_document_external_cause"])
            terminal = "document_external_cause_define_response"
            headline = (
                f"Ranking losses are spread across many pages "
                f"({isolation['evidence']}) — points to algorithm update, "
                "competitor SERP gains, or another external driver."
            )
            next_check = (
                "Check the Google algorithm-update log and top-competitor SERPs "
                "for recent changes. Document and assign owners."
            )

        return DiagnosticReport(
            triggered=True,
            session_delta_pct=session_delta_pct,
            session_delta_abs=session_delta_abs,
            session_direction=direction,
            trigger_note=trigger_note,
            concentration=conc["verdict"],
            concentration_evidence=conc["evidence"],
            concentration_theme=conc.get("theme", []),
            ranking=rank["verdict"],
            ranking_evidence=rank["evidence"],
            isolation=isolation["verdict"],
            isolation_evidence=isolation["evidence"],
            terminal_node=terminal,  # type: ignore[arg-type]
            headline=headline,
            recommended_next_check=next_check,
            visited_nodes=visited_nodes,
            gate_decisions=gate_decisions,
        )

    # --- Ranking branch C: stable → search interest → CTR ---
    # (The "unknown" case falls through to this branch too so we at least
    # render the impressions/CTR lens when ranking data is missing.)
    visited_nodes.append("check_search_interest")
    impr = _evaluate_impressions(pct_change_impressions=pct_change_impressions,
                                  impression_effect=impression_effect)
    gate_decisions.append(GateDecision(
        name="Did impressions decline between periods?",
        verdict=impr["verdict"],
        evidence=impr["evidence"],
        edge_label={"declined": "Yes", "stable": "No, impressions stable",
                    "grew": "No (impressions grew)", "unknown": "—"}[impr["verdict"]],
    ))
    visited_nodes.append("diamond_impressions")

    if impr["verdict"] == "declined":
        visited_nodes.append("terminal_reduced_search_interest")
        headline = (
            f"Rankings held but impressions fell "
            f"{_fmt_pct(pct_change_impressions)} — the decline is primarily a "
            "reduced-search-interest / macro demand story."
        )
        return DiagnosticReport(
            triggered=True,
            session_delta_pct=session_delta_pct,
            session_delta_abs=session_delta_abs,
            session_direction=direction,
            trigger_note=trigger_note,
            concentration=conc["verdict"],
            concentration_evidence=conc["evidence"],
            concentration_theme=conc.get("theme", []),
            ranking=rank["verdict"],
            ranking_evidence=rank["evidence"],
            impressions=impr["verdict"],
            impressions_evidence=impr["evidence"],
            terminal_node="attribute_to_reduced_search_interest",
            headline=headline,
            recommended_next_check=(
                "Pull Google Trends for the top themes and overlay with known "
                "seasonality curves (billing cycles, heat waves, winter storms)."
            ),
            visited_nodes=visited_nodes,
            gate_decisions=gate_decisions,
        )

    # --- Impressions stable → CTR gate ---
    visited_nodes.append("check_ctr_efficiency")
    ctr = _evaluate_ctr(pct_change_ctr=pct_change_ctr, ctr_effect=ctr_effect)
    gate_decisions.append(GateDecision(
        name="Did CTR fall?",
        verdict=ctr["verdict"],
        evidence=ctr["evidence"],
        edge_label={"fell": "Yes", "stable": "No", "rose": "No (CTR rose)",
                    "unknown": "—"}[ctr["verdict"]],
    ))
    visited_nodes.append("diamond_ctr")

    if ctr["verdict"] == "fell":
        visited_nodes.append("terminal_serp_click_loss")
        headline = (
            f"Rank + impressions held but CTR fell "
            f"{_fmt_pct(pct_change_ctr)} — SERP click-loss drivers "
            "(AI Overviews, ads, snippets) are compressing clicks."
        )
        return DiagnosticReport(
            triggered=True,
            session_delta_pct=session_delta_pct,
            session_delta_abs=session_delta_abs,
            session_direction=direction,
            trigger_note=trigger_note,
            concentration=conc["verdict"],
            concentration_evidence=conc["evidence"],
            concentration_theme=conc.get("theme", []),
            ranking=rank["verdict"],
            ranking_evidence=rank["evidence"],
            impressions=impr["verdict"],
            impressions_evidence=impr["evidence"],
            ctr=ctr["verdict"],
            ctr_evidence=ctr["evidence"],
            terminal_node="investigate_serp_click_loss_drivers",
            headline=headline,
            recommended_next_check=(
                "Pull top-losing queries from `d_1` and inspect the live SERPs "
                "for AI Overviews, expanded ads, and featured-snippet changes."
            ),
            visited_nodes=visited_nodes,
            gate_decisions=gate_decisions,
        )

    # --- Impressions stable AND CTR stable → macro / measurement ---
    visited_nodes.append("terminal_macro_measurement")
    headline = (
        "Rank, impressions, and CTR are all stable — the move isn't a "
        "ranking, demand, or SERP-efficiency story. Likely macro / "
        "measurement / traffic-mix (tagging, bot filtering, channel "
        "reclassification, or a cross-channel mix shift)."
    )
    return DiagnosticReport(
        triggered=True,
        session_delta_pct=session_delta_pct,
        session_delta_abs=session_delta_abs,
        session_direction=direction,
        trigger_note=trigger_note,
        concentration=conc["verdict"],
        concentration_evidence=conc["evidence"],
        concentration_theme=conc.get("theme", []),
        ranking=rank["verdict"],
        ranking_evidence=rank["evidence"],
        impressions=impr["verdict"],
        impressions_evidence=impr["evidence"],
        ctr=ctr["verdict"],
        ctr_evidence=ctr["evidence"],
        terminal_node="review_other_macro_measurement_factors",
        headline=headline,
        recommended_next_check=(
            "Check for tagging / bot-filter changes and the cross-channel mix "
            "(paid, direct, referral) for offsetting moves."
        ),
        visited_nodes=visited_nodes,
        gate_decisions=gate_decisions,
    )


# ---------------------------------------------------------------------------
# Individual gate evaluators
# ---------------------------------------------------------------------------


def _evaluate_concentration(movers: list[dict] | None) -> dict:
    if not movers:
        return {"verdict": "unknown",
                "evidence": "Page-type click-delta data unavailable.",
                "theme": []}
    total_abs = sum(abs(float(m.get("click_delta", 0))) for m in movers)
    if total_abs <= 0:
        return {"verdict": "no_concentration",
                "evidence": "Click delta is effectively zero across page types.",
                "theme": []}

    ranked = sorted(movers, key=lambda m: abs(float(m.get("click_delta", 0))),
                    reverse=True)
    top1_abs = abs(float(ranked[0]["click_delta"]))
    top1_share = top1_abs / total_abs
    top2_abs = abs(float(ranked[1]["click_delta"])) if len(ranked) > 1 else 0.0
    top1_over_top2 = top1_abs / top2_abs if top2_abs > 0 else float("inf")

    is_concentrated = (
        top1_share >= CONCENTRATION_TOP1_SHARE
        and top1_over_top2 >= CONCENTRATION_TOP1_OVER_TOP2
    )

    if is_concentrated:
        # Only name page types that individually carry a meaningful slice.
        # #1 is always in the theme. #2 is included only if it's itself
        # a material mover (≥ 20% of |Δ|), otherwise the theme is singular.
        theme = [str(ranked[0]["landing_page_type"])]
        if top2_abs / total_abs >= 0.20:
            theme.append(str(ranked[1]["landing_page_type"]))
        return {
            "verdict": "concentrated",
            "evidence": (
                f"Yes — **{ranked[0]['landing_page_type']}** alone drives "
                f"{top1_share * 100:.0f}% of the click delta "
                f"({top1_over_top2:.1f}× the next-largest mover)."
            ),
            "theme": theme,
        }
    return {
        "verdict": "no_concentration",
        "evidence": (
            f"No — top page-type (`{ranked[0]['landing_page_type']}`) only explains "
            f"{top1_share * 100:.0f}% of the click delta; the move is spread "
            "across the portfolio."
        ),
        "theme": [],
    }


def _evaluate_ranking(
    *,
    curr_rank: float | None,
    prior_rank: float | None,
    page_type_movers: list[dict] | None,
    page1_churn_pct: float | None,
    concentration_verdict: ConcentrationVerdict = "unknown",
    concentration_theme: list[str] | None = None,
    session_share_map: dict[str, float] | None = None,
) -> dict:
    """Hybrid ranking gate — fires on either weighted-rank slip or page-1 churn.

    Returns one of: specific_categories / many_types / stable / unknown.

    If Gate 1 already identified a `concentrated` theme, the ranking gate
    prefers `specific_categories` (because the click-loss is contained in
    that theme) even when several page types are directionally negative.
    """
    rank_delta: float | None = None
    if curr_rank is not None and prior_rank is not None:
        rank_delta = curr_rank - prior_rank  # positive = worse

    weighted_fired = rank_delta is not None and rank_delta >= RANK_THRESHOLD_POSITIONS
    churn_fired = page1_churn_pct is not None and page1_churn_pct >= PAGE1_CHURN_THRESHOLD

    if not weighted_fired and not churn_fired:
        if rank_delta is None and page1_churn_pct is None:
            return {
                "verdict": "unknown",
                "evidence": "Rank data unavailable for at least one window.",
            }
        bits: list[str] = []
        if rank_delta is not None:
            bits.append(
                f"weighted avg rank moved {rank_delta:+.2f} positions "
                f"({prior_rank:.1f} → {curr_rank:.1f})"
            )
        if page1_churn_pct is not None:
            bits.append(f"{page1_churn_pct * 100:.0f}% of page-1 queries fell off page 1")
        return {
            "verdict": "stable",
            "evidence": (
                "Stable — " + "; ".join(bits)
                + f" (below the {RANK_THRESHOLD_POSITIONS:.1f}-position / "
                  f"{PAGE1_CHURN_THRESHOLD * 100:.0f}% thresholds)."
            ),
        }

    # Ranking did drop — is it spread across many page types or concentrated?
    losing_types = [m for m in (page_type_movers or [])
                    if float(m.get("click_delta", 0)) < 0]
    breadth = len(losing_types)

    bits: list[str] = []
    if weighted_fired and rank_delta is not None:
        bits.append(
            f"weighted avg rank **worsened {rank_delta:+.2f} positions** "
            f"({prior_rank:.1f} → {curr_rank:.1f})"
        )
    if churn_fired and page1_churn_pct is not None:
        bits.append(
            f"**{page1_churn_pct * 100:.0f}%** of previously page-1 queries "
            "fell off page 1"
        )
    evidence_core = " and ".join(bits)

    # Gate-1 concentration overrides breadth heuristic: if a single theme
    # is driving the click loss, treat the ranking drop as scoped to that
    # theme even when more than 2 page types show some red ink.
    if concentration_verdict == "concentrated":
        theme = list(concentration_theme or [])
        names = ", ".join(f"`{t}`" for t in theme) if theme else "the impacted theme"
        return {
            "verdict": "specific_categories",
            "evidence": (
                f"Yes, for specific keyword categories — {evidence_core}; "
                f"losses concentrated in {names} (Gate 1 concentration)."
            ),
        }

    if breadth >= BROAD_PAGE_TYPE_COUNT:
        # Add session-volume context so the evidence reflects business impact.
        share_map = session_share_map or {}
        if share_map:
            losing_with_share = sorted(
                losing_types,
                key=lambda m: share_map.get(str(m.get("landing_page_type", "")), 0),
                reverse=True,
            )
            top_losers = losing_with_share[:3]
            share_bits = []
            total_losing_share = 0.0
            for m in top_losers:
                lpt = str(m.get("landing_page_type", ""))
                share = share_map.get(lpt, 0)
                total_losing_share += share
                share_bits.append(f"`{lpt}` ({share * 100:.0f}% of sessions)")
            volume_note = (
                f" The highest-volume impacted page types are: "
                + ", ".join(share_bits) + "."
            )
        else:
            volume_note = ""
        return {
            "verdict": "many_types",
            "evidence": (
                f"Yes, across many keyword types — {evidence_core}; losses show "
                f"up in {breadth} page types.{volume_note}"
            ),
        }
    if breadth >= 1:
        names = ", ".join(f"`{m['landing_page_type']}`" for m in losing_types)
        return {
            "verdict": "specific_categories",
            "evidence": (
                f"Yes, for specific keyword categories — {evidence_core}; losses "
                f"concentrated in {names}."
            ),
        }
    # Fired the threshold but we don't have page-type breakdown to classify.
    return {
        "verdict": "many_types",
        "evidence": (
            f"Yes — {evidence_core}. Page-type breakdown unavailable, so "
            "scope is ambiguous; treating as broad by default."
        ),
    }


def _evaluate_isolation(page_type_movers: list[dict] | None) -> dict:
    losers = [m for m in (page_type_movers or [])
              if float(m.get("click_delta", 0)) < 0]
    if not losers:
        return {"verdict": "n/a",
                "evidence": "No page types showed a click decline."}
    n = len(losers)
    names = ", ".join(f"`{m['landing_page_type']}`" for m in losers[:3])
    if n <= FEW_PAGES_MAX:
        return {"verdict": "few_pages",
                "evidence": f"Few pages only — {n} page type(s) lost clicks ({names})."}
    return {"verdict": "many_pages",
            "evidence": f"Many pages — {n} page types lost clicks (top: {names})."}


def _evaluate_impressions(
    *,
    pct_change_impressions: float | None,
    impression_effect: float | None,
) -> dict:
    if pct_change_impressions is None:
        return {"verdict": "unknown", "evidence": "Impression data unavailable."}
    if abs(pct_change_impressions) < IMPRESSION_MOVE_THRESHOLD_PCT:
        return {
            "verdict": "stable",
            "evidence": (
                f"Impressions moved only {_fmt_pct(pct_change_impressions)} "
                f"(below ±{IMPRESSION_MOVE_THRESHOLD_PCT * 100:.0f}%)."
            ),
        }
    if pct_change_impressions < 0:
        return {
            "verdict": "declined",
            "evidence": (
                f"Impressions declined {_fmt_pct(pct_change_impressions)} "
                f"(~{_fmt_clicks(impression_effect)} clicks via the impression effect)."
            ),
        }
    return {
        "verdict": "grew",
        "evidence": (
            f"Impressions grew {_fmt_pct(pct_change_impressions)} "
            f"(~{_fmt_clicks(impression_effect)} clicks via the impression effect)."
        ),
    }


def _evaluate_ctr(
    *,
    pct_change_ctr: float | None,
    ctr_effect: float | None,
) -> dict:
    if pct_change_ctr is None:
        return {"verdict": "unknown", "evidence": "CTR data unavailable."}
    if abs(pct_change_ctr) < CTR_MOVE_THRESHOLD_PCT:
        return {
            "verdict": "stable",
            "evidence": (
                f"CTR moved only {_fmt_pct(pct_change_ctr)} "
                f"(below ±{CTR_MOVE_THRESHOLD_PCT * 100:.0f}%)."
            ),
        }
    if pct_change_ctr < 0:
        return {
            "verdict": "fell",
            "evidence": (
                f"CTR fell {_fmt_pct(pct_change_ctr)} "
                f"(~{_fmt_clicks(ctr_effect)} clicks via the CTR effect)."
            ),
        }
    return {
        "verdict": "rose",
        "evidence": (
            f"CTR rose {_fmt_pct(pct_change_ctr)} "
            f"(~{_fmt_clicks(ctr_effect)} clicks via the CTR effect)."
        ),
    }


def _build_headline_specific_categories(
    *,
    direction: str,
    session_delta_pct: float | None,
    ranking_theme: list[str],
    rank_evidence: str,
) -> str:
    theme_str = (
        ", ".join(f"`{t}`" for t in ranking_theme)
        if ranking_theme else "the affected segment"
    )
    return (
        f"Ranking losses concentrated in {theme_str} — refresh content and "
        "on-page elements for those keyword categories to recover lost rank."
    )


# ---------------------------------------------------------------------------
# Graphviz rendering
# ---------------------------------------------------------------------------

# Node layout for the manager's flowchart. `id` → (label, shape).
_FLOWCHART_NODES: list[tuple[str, str, str]] = [
    ("start",                          "SEO sessions changed",                          "box"),
    ("initial_check",                  "Check whether change was driven by\nspecific pages / domains",  "box"),
    ("diamond_concentration",          "Outsized impact from specific\npage types / domains?",          "diamond"),
    ("identify_theme",                 "Identify theme across\nimpacted pages\n(geo, product, blog …)", "box"),
    ("review_sitewide",                "Review broader sitewide\nkeyword & page patterns",               "box"),
    ("check_keyword_rankings",         "Check keyword-level\nranking changes for\nimpacted pages/theme", "box"),
    ("diamond_rank_drops",             "Were there keyword\nranking drops?",                             "diamond"),
    ("terminal_optimize_keyword_categories",
                                       "Optimize for affected\nkeyword categories",                     "box"),
    ("diamond_isolation",              "Ranking loss isolated to\na few pages or many pages?",          "diamond"),
    ("review_content_outdated",        "Review whether content is\noutdated / less competitive",        "box"),
    ("terminal_refresh_optimize_content",
                                       "Refresh and broadly\noptimize content",                         "box"),
    ("investigate_external_forces",    "Investigate external forces\n(algorithm, competitor, SERP)",    "box"),
    ("terminal_document_external_cause",
                                       "Document external cause &\ndefine response plan",               "box"),
    ("check_search_interest",          "Check search interest\n(GSC impressions, page-1 / top-5 kws)",  "box"),
    ("diamond_impressions",            "Did impressions decline\nbetween periods?",                     "diamond"),
    ("terminal_reduced_search_interest",
                                       "Attribute decline to reduced\nsearch interest / macro demand",  "box"),
    ("check_ctr_efficiency",           "Check CTR / click efficiency",                                  "box"),
    ("diamond_ctr",                    "Did CTR fall?",                                                 "diamond"),
    ("terminal_serp_click_loss",       "Investigate SERP click-loss\ndrivers (AIO, ads, snippets)",     "box"),
    ("terminal_macro_measurement",     "Review other macro /\nmeasurement / mix factors",               "box"),
    ("session_stable",                 "Sessions stable — no walk",                                     "box"),
]

_FLOWCHART_EDGES: list[tuple[str, str, str]] = [
    ("start", "initial_check", ""),
    ("initial_check", "diamond_concentration", ""),
    ("diamond_concentration", "identify_theme", "Yes"),
    ("diamond_concentration", "review_sitewide", "No clear\nconcentration"),
    ("identify_theme", "check_keyword_rankings", ""),
    ("review_sitewide", "check_keyword_rankings", ""),
    ("check_keyword_rankings", "diamond_rank_drops", ""),
    ("diamond_rank_drops", "terminal_optimize_keyword_categories",
                                                "Yes, for specific\nkeyword categories"),
    ("diamond_rank_drops", "diamond_isolation", "Yes, across many\nkeyword types"),
    ("diamond_rank_drops", "check_search_interest",
                                                "No notable keyword-\nlevel ranking drops"),
    ("diamond_isolation", "review_content_outdated",   "Few pages only"),
    ("diamond_isolation", "investigate_external_forces", "Many pages"),
    ("review_content_outdated",     "terminal_refresh_optimize_content",   ""),
    ("investigate_external_forces", "terminal_document_external_cause",    ""),
    ("check_search_interest",       "diamond_impressions",                 ""),
    ("diamond_impressions",         "terminal_reduced_search_interest",    "Yes"),
    ("diamond_impressions",         "check_ctr_efficiency",                "No, impressions\nstable"),
    ("check_ctr_efficiency",        "diamond_ctr",                         ""),
    ("diamond_ctr",                 "terminal_serp_click_loss",            "Yes"),
    ("diamond_ctr",                 "terminal_macro_measurement",          "No"),
]


def _build_graphviz_dot(visited: list[str]) -> str:
    """Emit a DOT string highlighting the visited path in orange."""
    visited_set = set(visited)
    lines: list[str] = [
        "digraph flowchart {",
        '  rankdir=TB;',
        '  bgcolor="white";',
        '  node [fontname="Helvetica", fontsize=10, margin="0.15,0.08"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#9AA0A6"];',
    ]
    for node_id, label, shape in _FLOWCHART_NODES:
        if node_id in visited_set:
            # Terminals get a stronger orange; traversed diamonds/boxes get a
            # lighter fill so the *final* destination reads as the answer.
            if node_id.startswith("terminal_") or node_id == "session_stable":
                fillcolor = "#F8961E"    # strong orange
                fontcolor = "white"
                color = "#CA6E0A"
                penwidth = "2.5"
            else:
                fillcolor = "#FFD8A8"    # soft orange
                fontcolor = "#1A1A1A"
                color = "#CA6E0A"
                penwidth = "1.5"
        else:
            fillcolor = "#F1F3F4"
            fontcolor = "#5F6368"
            color = "#BDC1C6"
            penwidth = "1.0"
        escaped = label.replace('"', '\\"').replace("\n", "\\n")
        lines.append(
            f'  {node_id} [shape={shape}, style="filled,rounded", label="{escaped}", '
            f'fillcolor="{fillcolor}", fontcolor="{fontcolor}", color="{color}", '
            f'penwidth="{penwidth}"];'
        )

    # Edges — highlight the ones *between* visited nodes (both endpoints visited).
    for src, dst, lab in _FLOWCHART_EDGES:
        on_path = src in visited_set and dst in visited_set
        attrs: list[str] = []
        if lab:
            attrs.append(f'label="{lab.replace(chr(10), chr(92) + "n")}"')
        if on_path:
            attrs.append('color="#F8961E"')
            attrs.append('penwidth="2.5"')
            attrs.append('fontcolor="#1A1A1A"')
        else:
            attrs.append('color="#BDC1C6"')
        lines.append(f'  {src} -> {dst} [' + ", ".join(attrs) + '];')

    lines.append("}")
    return "\n".join(lines)
