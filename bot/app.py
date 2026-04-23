"""
Slack Bot entry point using Bolt + Socket Mode.

Handles @mentions and threaded replies, routes them through the
OpenAI agent loop, and posts responses back to Slack.

Usage:
    python -m bot.app
"""

import io
import logging
import os
import re
import time
import threading

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from bot.agent import run_agent
from bot.charts import render_pacing_table, render_waterfall, render_tof_chart
from bot.thread_store import ThreadStore

load_dotenv(dotenv_path=".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")

app = App(token=SLACK_BOT_TOKEN)
threads = ThreadStore()

THINKING_EMOJI = "hourglass_flowing_sand"
DONE_EMOJI = "white_check_mark"
BOT_USER_ID: str | None = None

# Slack messages have a 4000-char limit per block; split if needed.
SLACK_MAX_CHARS = 3900


def _resolve_bot_id():
    """Fetch the bot's own user ID so we can detect DMs vs channels."""
    global BOT_USER_ID
    if BOT_USER_ID is None:
        try:
            resp = app.client.auth_test()
            BOT_USER_ID = resp.get("user_id", "")
            logger.info("Bot user ID: %s", BOT_USER_ID)
        except Exception:
            logger.warning("Could not resolve bot user ID")


def _add_reaction(channel: str, ts: str, emoji: str):
    """Add a reaction, silently ignoring errors (e.g. in DMs)."""
    try:
        app.client.reactions_add(channel=channel, name=emoji, timestamp=ts)
    except Exception:
        pass


def _swap_reaction(channel: str, ts: str, old_emoji: str, new_emoji: str):
    """Remove old reaction and add new one, ignoring errors."""
    try:
        app.client.reactions_remove(channel=channel, name=old_emoji, timestamp=ts)
    except Exception:
        pass
    _add_reaction(channel, ts, new_emoji)


def _strip_mention(text: str) -> str:
    """Remove the <@BOT_ID> mention prefix from message text."""
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def _markdown_to_slack(text: str) -> str:
    """Convert common Markdown patterns to Slack mrkdwn format.

    Slack uses its own formatting:
      *bold*  (not **bold**)
      _italic_  (not *italic* — but we leave single * as bold since
                 that's what we instruct the LLM to use)
      ~strike~  (not ~~strike~~)
      No ### headers — convert to *bold line*
      Bullet lists with • instead of -  (optional, - works in Slack)
    """
    # Headers: ### Title → *Title*  (bold on its own line)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)

    # Bold: **text** → *text*  (must run before italic)
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)

    # Italic: __text__ → _text_
    text = re.sub(r"__(.+?)__", r"_\1_", text)

    # Strikethrough: ~~text~~ → ~text~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)

    # Inline code is fine in Slack (`code`) — no change needed
    # Code blocks (```lang\n...\n```) are fine in Slack — no change needed

    # Collapse duplicate bold markers that can appear after conversion
    # e.g. **bold** → *bold* is correct, but ***bold*** → **bold** needs fixing
    text = re.sub(r"\*{2,}(.+?)\*{2,}", r"*\1*", text)

    # Horizontal rules: --- or *** → ———
    text = re.sub(r"^[-*_]{3,}\s*$", "———", text, flags=re.MULTILINE)

    return text


def _split_for_slack(text: str) -> list[str]:
    """Split long messages into chunks that fit within Slack's limit."""
    if len(text) <= SLACK_MAX_CHARS:
        return [text]

    chunks = []
    while text:
        if len(text) <= SLACK_MAX_CHARS:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, SLACK_MAX_CHARS)
        if split_at == -1:
            split_at = SLACK_MAX_CHARS
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def _upload_chart(channel: str, thread_ts: str, png_bytes: bytes, filename: str, title: str):
    """Upload a chart image to Slack in the given thread."""
    try:
        app.client.files_upload_v2(
            channel=channel,
            thread_ts=thread_ts,
            file=io.BytesIO(png_bytes),
            filename=filename,
            title=title,
        )
    except Exception:
        logger.exception("Failed to upload chart %s", filename)


def _generate_and_upload_charts(artifacts: dict, channel: str, thread_ts: str):
    """Generate chart images from tool artifacts and upload them to Slack."""
    pacing_df = artifacts.get("pacing_df")
    gsc_df = artifacts.get("gsc_df")
    month = artifacts.get("_month")
    as_of = artifacts.get("_as_of")

    if pacing_df is not None:
        try:
            png = render_pacing_table(pacing_df, month=month, as_of=as_of)
            if png:
                _upload_chart(channel, thread_ts, png, "pacing_snapshot.png", "SEO Pacing Snapshot")
        except Exception:
            logger.exception("Failed to render pacing table")

        try:
            png = render_waterfall(pacing_df, month=month, as_of=as_of)
            if png:
                _upload_chart(channel, thread_ts, png, "revenue_waterfall.png", "Revenue Waterfall")
        except Exception:
            logger.exception("Failed to render waterfall chart")

    if gsc_df is not None:
        try:
            png = render_tof_chart(gsc_df, month=month, as_of=as_of)
            if png:
                _upload_chart(channel, thread_ts, png, "tof_performance.png", "Top-of-Funnel Performance")
        except Exception:
            logger.exception("Failed to render TOF chart")


def _handle_message(user_text: str, thread_ts: str, channel: str, say):
    """Process a user message through the agent and reply in-thread."""
    threads.add_message(thread_ts, "user", user_text)
    history = threads.get_messages(thread_ts)

    logger.info(
        "Processing message in thread %s (%d messages in history)",
        thread_ts,
        len(history),
    )

    artifacts = {}
    try:
        response_text, artifacts = run_agent(history)
    except Exception:
        logger.exception("Agent failed for thread %s", thread_ts)
        response_text = (
            "Sorry, I ran into an error processing that request. "
            "Please try again or rephrase your question."
        )

    threads.add_message(thread_ts, "assistant", response_text)

    logger.info("Agent returned %d artifacts: %s", len(artifacts), list(artifacts.keys()))
    _generate_and_upload_charts(artifacts, channel, thread_ts)

    slack_text = _markdown_to_slack(response_text)
    logger.info("Markdown converted, posting %d chars to Slack", len(slack_text))
    chunks = _split_for_slack(slack_text)
    for chunk in chunks:
        say(text=chunk, thread_ts=thread_ts)


@app.event("app_mention")
def handle_mention(event, say):
    """Respond to @bot mentions in channels -- starts or continues a thread."""
    user_text = _strip_mention(event.get("text", ""))
    if not user_text:
        say(
            text="Hi! Ask me about SEO pacing, organic performance, or search diagnostics.",
            thread_ts=event.get("thread_ts", event["ts"]),
        )
        return

    thread_ts = event.get("thread_ts", event["ts"])
    channel = event["channel"]

    _add_reaction(channel, event["ts"], THINKING_EMOJI)
    _handle_message(user_text, thread_ts, channel, say)
    _swap_reaction(channel, event["ts"], THINKING_EMOJI, DONE_EMOJI)


def _is_dm(channel_type: str) -> bool:
    return channel_type in ("im", "mpim")


@app.event("message")
def handle_message_event(event, say):
    """
    Handle all message events:
    - DMs: treat every message as directed at the bot
    - Channels: only respond in threads the bot already participates in
    """
    if event.get("subtype"):
        return
    if "bot_id" in event:
        return

    user_text = _strip_mention(event.get("text", "")).strip()
    if not user_text:
        return

    channel = event["channel"]
    channel_type = event.get("channel_type", "")

    if _is_dm(channel_type):
        thread_ts = event.get("thread_ts", event["ts"])
        _add_reaction(channel, event["ts"], THINKING_EMOJI)
        _handle_message(user_text, thread_ts, channel, say)
        _swap_reaction(channel, event["ts"], THINKING_EMOJI, DONE_EMOJI)
        return

    thread_ts = event.get("thread_ts")
    if not thread_ts:
        return
    if not threads.get_messages(thread_ts):
        return

    _add_reaction(channel, event["ts"], THINKING_EMOJI)
    _handle_message(user_text, thread_ts, channel, say)
    _swap_reaction(channel, event["ts"], THINKING_EMOJI, DONE_EMOJI)


def _cleanup_loop():
    """Periodically clean up expired thread histories."""
    while True:
        time.sleep(600)
        removed = threads.cleanup_expired()
        if removed:
            logger.info("Cleaned up %d expired threads (%d active)", removed, threads.active_threads)


def main():
    if not SLACK_BOT_TOKEN:
        raise RuntimeError("SLACK_BOT_TOKEN not set in .env")
    if not SLACK_APP_TOKEN:
        raise RuntimeError("SLACK_APP_TOKEN not set in .env")

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()

    _resolve_bot_id()

    logger.info("Starting SEO Reporting Bot in Socket Mode (v2 — charts + slack formatting)...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()


if __name__ == "__main__":
    main()
