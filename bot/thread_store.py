"""
In-memory conversation thread store for the Slack bot.

Maps Slack thread_ts → list of messages [{role, content}].
Includes TTL-based expiration so old threads don't leak memory.
"""

import threading
import time
from typing import Any

DEFAULT_TTL_SECONDS = 4 * 60 * 60  # 4 hours
MAX_MESSAGES_PER_THREAD = 50


class ThreadStore:
    """Thread-safe in-memory store for conversation histories."""

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS):
        self._threads: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl

    def get_messages(self, thread_ts: str) -> list[dict]:
        """Return the message history for a thread, or empty list."""
        with self._lock:
            entry = self._threads.get(thread_ts)
            if entry is None:
                return []
            entry["last_access"] = time.time()
            return list(entry["messages"])

    def add_message(self, thread_ts: str, role: str, content: str) -> None:
        """Append a message to a thread's history."""
        with self._lock:
            if thread_ts not in self._threads:
                self._threads[thread_ts] = {
                    "messages": [],
                    "created": time.time(),
                    "last_access": time.time(),
                }
            entry = self._threads[thread_ts]
            entry["messages"].append({"role": role, "content": content})
            entry["last_access"] = time.time()

            if len(entry["messages"]) > MAX_MESSAGES_PER_THREAD:
                entry["messages"] = entry["messages"][-MAX_MESSAGES_PER_THREAD:]

    def cleanup_expired(self) -> int:
        """Remove threads older than TTL. Returns count of removed threads."""
        now = time.time()
        removed = 0
        with self._lock:
            expired = [
                ts for ts, entry in self._threads.items()
                if now - entry["last_access"] > self._ttl
            ]
            for ts in expired:
                del self._threads[ts]
                removed += 1
        return removed

    @property
    def active_threads(self) -> int:
        with self._lock:
            return len(self._threads)
