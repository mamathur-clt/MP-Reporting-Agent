"""
Shared application context for the v2 tabbed Streamlit app.

`AppContext` packages every piece of sidebar state and pre-loaded data a tab
might need: selected KPI/filters, resolved current+prior periods, raw and
filtered session DataFrames, finance pacing DataFrames, and (lazily) the
common decomposition artefacts used by several sections of the Overview tab.

Tabs should never re-fetch the sidebar inputs themselves; they take a
fully-populated `AppContext` and render from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd


@dataclass
class AppContext:
    # Sidebar selections
    kpi_key: str
    time_mode: str
    channel_filter: list[str]
    website_filter: list[str]
    available_channels: list[str]
    available_websites: list[str]

    # Resolved periods
    curr_start: date
    curr_end: date
    prior_start: date
    prior_end: date
    query_start: date
    query_end: date

    # Session-level data (Tier 2)
    df_all: pd.DataFrame
    df_filtered: pd.DataFrame
    df_current: pd.DataFrame
    df_prior: pd.DataFrame

    # Finance data (Tier 1). May be None if the finance fetch failed.
    finance_df: Optional[pd.DataFrame] = None
    plan_df: Optional[pd.DataFrame] = None

    # Cached derived artefacts — populated on demand by tabs.
    cache: dict = field(default_factory=dict)

    @property
    def period_label(self) -> str:
        return f"{self.curr_start.strftime('%-m/%-d')} – {self.curr_end.strftime('%-m/%-d/%y')}"

    @property
    def effective_channels(self) -> list[str]:
        """Channels that should appear in finance summaries: honour the user
        filter when set, otherwise fall back to every channel present in the
        loaded data."""
        return self.channel_filter if self.channel_filter else self.available_channels
