"""
OpenAI agent loop with tool calling for the Slack bot.

Adapted from app/narrative.py's run_analyst_chat, but decoupled from
Streamlit and designed for multi-thread Slack conversations.
"""

import json
import logging
import os

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from bot.prompts.seo import build_seo_system_prompt
from bot.tools import seo as seo_tools
from bot.tools import sql as sql_tool

load_dotenv(dotenv_path=".env", override=True)

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
MODEL = "gpt-4o"
MAX_TOOL_ROUNDS = 5
MAX_RESPONSE_TOKENS = 3500

_ARTIFACT_TOOL_MAP = {
    "run_seo_pacing": "pacing_df",
    "run_gsc_summary": "gsc_df",
}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _all_tool_definitions() -> list[dict]:
    """Combine all available tool definitions."""
    return seo_tools.TOOL_DEFINITIONS + [sql_tool.TOOL_DEFINITION]


def _dispatch_tool(
    name: str,
    args_json: str,
    artifacts: dict[str, pd.DataFrame],
) -> str:
    """Route a tool call to the right executor, return result text.

    If the tool produces a DataFrame, store it in *artifacts* under
    the key defined in _ARTIFACT_TOOL_MAP.
    """
    if name in seo_tools._EXECUTORS:
        result_str, explanation, df = seo_tools.execute(name, args_json)
        logger.info("Tool %s: %s", name, explanation)
        artifact_key = _ARTIFACT_TOOL_MAP.get(name)
        if artifact_key and df is not None:
            artifacts[artifact_key] = df
        if name in _ARTIFACT_TOOL_MAP:
            try:
                args = json.loads(args_json)
                artifacts.setdefault("_month", args.get("month"))
                artifacts.setdefault("_as_of", args.get("as_of_date"))
            except (json.JSONDecodeError, AttributeError):
                pass
        return result_str

    if name == "run_databricks_sql":
        result_str, explanation = sql_tool.execute(args_json)
        logger.info("Tool %s: %s", name, explanation)
        return result_str

    return f"Unknown tool: {name}"


def run_agent(
    messages: list[dict],
    system_prompt: str | None = None,
) -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Run the agent loop: send messages to OpenAI, execute any tool calls,
    and return the final text response plus any DataFrame artifacts.

    Args:
        messages: Conversation history as [{role, content}, ...].
        system_prompt: Override system prompt. Defaults to SEO prompt.

    Returns:
        (response_text, artifacts) where artifacts maps names like
        "pacing_df" / "gsc_df" to their DataFrames.
    """
    if system_prompt is None:
        system_prompt = build_seo_system_prompt()

    client = _get_client()
    tools = _all_tool_definitions()
    artifacts: dict[str, pd.DataFrame] = {}

    full_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    for round_num in range(MAX_TOOL_ROUNDS):
        logger.info("Agent round %d/%d", round_num + 1, MAX_TOOL_ROUNDS)

        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            tools=tools,
            temperature=0.3,
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return msg.content or "", artifacts

        full_messages.append(msg)
        for tc in msg.tool_calls:
            logger.info(
                "Tool call: %s(%s)",
                tc.function.name,
                tc.function.arguments[:120],
            )
            try:
                result_str = _dispatch_tool(tc.function.name, tc.function.arguments, artifacts)
            except Exception as e:
                logger.exception("Tool %s failed", tc.function.name)
                result_str = f"Tool error: {type(e).__name__}: {e}"

            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    last_content = response.choices[0].message.content if response else ""
    return (
        last_content or "Analysis could not be completed within the allowed tool rounds.",
        artifacts,
    )
