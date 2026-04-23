"""
Ask the Analyst tab — the v1 Section 7 chat, promoted to its own tab.

The chat reuses whatever the Overview tab already computed
(`summary`, `top_drivers`, `init_current`, `impact_current`, `impact_prior`)
by reading them from `AppContext.cache`. If those keys are missing (e.g. the
user navigated straight to this tab), we recompute on the fly so the tab is
self-contained.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.analyst_tools import (
    TOOL_DEFINITIONS,
    build_analyst_system_prompt,
    build_schema_context,
    make_tool_executor,
)
from app.app_context import AppContext
from app.config import DRIVER_DIMENSIONS
from app.decomposition import (
    analyze_initiatives,
    compute_initiative_impact,
    decompose_all_dimensions,
    rank_top_drivers,
)
from app.kpi_engine import compute_funnel_table, compute_kpi_summary
from app.narrative import build_chat_system_prompt, run_analyst_chat


def _ensure_context_artefacts(ctx: AppContext) -> dict:
    """Reuse Overview-tab artefacts when present; recompute otherwise."""
    cache = ctx.cache
    missing = any(
        k not in cache
        for k in (
            "overview_summary",
            "overview_top_drivers",
            "overview_init_current",
            "overview_impact_current",
            "overview_impact_prior",
        )
    )
    if not missing:
        return cache

    organic_only = bool(ctx.channel_filter) and all(
        c.lower() == "organic" for c in ctx.channel_filter
    )
    driver_dims = [
        d for d in DRIVER_DIMENSIONS if d != "landing_page_type" or organic_only
    ]
    summary = compute_kpi_summary(ctx.df_current, ctx.df_prior, ctx.kpi_key)
    decomp_results = decompose_all_dimensions(
        ctx.df_current, ctx.df_prior, ctx.kpi_key, driver_dims
    )
    top_drivers = rank_top_drivers(decomp_results, n=8)
    init_current = analyze_initiatives(ctx.df_current, ctx.kpi_key)
    impact_current = compute_initiative_impact(ctx.df_current, ctx.kpi_key)
    impact_prior = compute_initiative_impact(ctx.df_prior, ctx.kpi_key)

    cache.update(
        overview_summary=summary,
        overview_top_drivers=top_drivers,
        overview_init_current=init_current,
        overview_impact_current=impact_current,
        overview_impact_prior=impact_prior,
    )
    return cache


def _funnel_text(ctx: AppContext) -> str:
    try:
        funnel = compute_funnel_table(ctx.df_current, ctx.df_prior)
        fc = funnel.copy()
        fc["current_rate"] = (funnel["current_rate"] * 100).round(2).astype(str) + "%"
        fc["prior_rate"] = (funnel["prior_rate"] * 100).round(2).astype(str) + "%"
        fc["delta"] = (funnel["delta"] * 100).map(lambda x: f"{x:+.2f}pp")
        return fc[["kpi", "current_rate", "prior_rate", "delta"]].to_string(index=False)
    except Exception:
        return ""


def render(ctx: AppContext) -> None:
    st.title("Ask the Analyst")
    st.caption(
        "Chat with GPT-4o — it can query the loaded session and finance data "
        "or run SQL against Databricks to answer ad-hoc questions."
    )

    artefacts = _ensure_context_artefacts(ctx)
    summary = artefacts["overview_summary"]
    top_drivers = artefacts["overview_top_drivers"]
    init_current = artefacts["overview_init_current"]
    impact_current = artefacts["overview_impact_current"]
    impact_prior = artefacts["overview_impact_prior"]

    chat_system_prompt = build_chat_system_prompt(
        summary,
        top_drivers,
        init_current,
        funnel_table_text=_funnel_text(ctx),
        initiative_impact=impact_current if not impact_current.empty else None,
        initiative_impact_prior=impact_prior if not impact_prior.empty else None,
    )

    schema_ctx = build_schema_context(ctx.df_all, ctx.finance_df)
    filter_parts = []
    if ctx.channel_filter:
        filter_parts.append(f"Channels: {', '.join(ctx.channel_filter)}")
    if ctx.website_filter:
        filter_parts.append(f"Websites: {', '.join(ctx.website_filter)}")
    analyst_prompt = build_analyst_system_prompt(
        chat_system_prompt,
        schema_ctx,
        current_filters="; ".join(filter_parts) if filter_parts else "",
    )
    tool_executor = make_tool_executor(ctx.df_all, ctx.finance_df)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            for tr in msg.get("tool_results", []):
                with st.expander(f"Data: {tr['explanation']}", expanded=False):
                    if isinstance(tr.get("result_obj"), pd.DataFrame):
                        st.dataframe(tr["result_obj"], use_container_width=True, hide_index=True)
                    elif tr.get("result_str"):
                        st.code(tr["result_str"])
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the data…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                try:
                    final_text, tool_results = run_analyst_chat(
                        st.session_state.chat_messages,
                        analyst_prompt,
                        TOOL_DEFINITIONS,
                        tool_executor,
                    )
                except Exception as e:
                    final_text = f"Error: {e}"
                    tool_results = []

            for tr in tool_results:
                with st.expander(f"Data: {tr['explanation']}", expanded=True):
                    if isinstance(tr.get("result_obj"), pd.DataFrame):
                        st.dataframe(tr["result_obj"], use_container_width=True, hide_index=True)
                    elif tr.get("result_str"):
                        st.code(tr["result_str"])

            st.markdown(final_text)
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": final_text,
                "tool_results": [
                    {
                        "explanation": tr["explanation"],
                        "result_str": tr["result_str"],
                        "result_obj": tr["result_obj"],
                    }
                    for tr in tool_results
                ],
            })
