"""
Time period resolution: maps a comparison mode (WoW, WTD, MoM, Custom)
into (current_start, current_end, prior_start, prior_end) date tuples
that can be used to slice the session DataFrame.
"""

from datetime import date, timedelta
import calendar


def _last_monday(ref: date) -> date:
    """Return the most recent Monday on or before *ref*."""
    return ref - timedelta(days=ref.weekday())


def resolve_periods(
    mode: str,
    ref_date: date | None = None,
    custom_current: tuple[date, date] | None = None,
    custom_prior: tuple[date, date] | None = None,
) -> tuple[date, date, date, date]:
    """
    Returns (current_start, current_end, prior_start, prior_end).
    All boundaries are inclusive.
    """
    today = ref_date or date.today()

    if mode == "WoW":
        # Current = last full week (Mon-Sun); Prior = the week before that
        last_sun = today - timedelta(days=today.weekday() + 1)
        curr_start = last_sun - timedelta(days=6)
        curr_end = last_sun
        prior_end = curr_start - timedelta(days=1)
        prior_start = prior_end - timedelta(days=6)

    elif mode == "WTD (Tue review)":
        # Current = yesterday (Monday); Prior = Monday of prior week
        curr_start = _last_monday(today - timedelta(days=1))
        curr_end = curr_start  # single day
        prior_start = curr_start - timedelta(weeks=1)
        prior_end = prior_start

    elif mode == "WTD (Fri review)":
        # Current = Mon-Thu of this week; Prior = Mon-Thu of last week
        mon = _last_monday(today)
        curr_start = mon
        curr_end = mon + timedelta(days=3)  # Thursday
        prior_start = mon - timedelta(weeks=1)
        prior_end = prior_start + timedelta(days=3)

    elif mode == "MoM":
        first_of_month = today.replace(day=1)
        prior_month_end = first_of_month - timedelta(days=1)
        prior_month_start = prior_month_end.replace(day=1)
        curr_start = prior_month_start + timedelta(
            days=calendar.monthrange(prior_month_start.year, prior_month_start.month)[1]
        )
        curr_end = today - timedelta(days=1)
        prior_start = prior_month_start
        prior_end = prior_month_end

    elif mode == "Custom":
        if custom_current and custom_prior:
            curr_start, curr_end = custom_current
            prior_start, prior_end = custom_prior
        else:
            raise ValueError("Custom mode requires both custom_current and custom_prior")
    else:
        raise ValueError(f"Unknown time mode: {mode}")

    return curr_start, curr_end, prior_start, prior_end
