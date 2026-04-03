"""
Deterministic datetime extraction from natural-language strings.
Returns an ISO-8601 string or None.  No AI involved.
"""
import re
from datetime import datetime, timedelta, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Patterns ordered from most to least specific
_TIME_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)
_RELATIVE_RE = re.compile(
    r"\bin\s+(\d+)\s+(minute|hour|day)s?",
    re.IGNORECASE,
)
_NEXT_WEEKDAY_RE = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    re.IGNORECASE,
)
_THIS_WEEKDAY_RE = re.compile(
    r"\b(this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    re.IGNORECASE,
)

_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _apply_time(base: datetime, text: str) -> datetime:
    """Overlay HH:MM from text onto base date if a time token is found."""
    m = _TIME_RE.search(text)
    if not m:
        return base.replace(hour=9, minute=0, second=0, microsecond=0)

    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    meridiem = (m.group(3) or "").lower()

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _next_occurrence(weekday_name: str, skip_current: bool = False) -> datetime:
    target = _WEEKDAY_MAP[weekday_name.lower()]
    now = _now()
    days_ahead = target - now.weekday()
    if days_ahead <= 0 or (days_ahead == 0 and skip_current):
        days_ahead += 7
    return now + timedelta(days=days_ahead)


def extract_datetime(text: str) -> Optional[str]:
    """
    Returns ISO-8601 UTC string if a date/time reference is detected,
    otherwise returns None.
    """
    lower = text.lower()
    now = _now()

    # "in X minutes/hours/days"
    m = _RELATIVE_RE.search(lower)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        delta = {
            "minute": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "day": timedelta(days=amount),
        }[unit]
        return (now + delta).isoformat()

    # "next <weekday>"
    m = _NEXT_WEEKDAY_RE.search(lower)
    if m:
        base = _next_occurrence(m.group(1), skip_current=True)
        return _apply_time(base, lower).isoformat()

    # "today"
    if re.search(r"\btoday\b", lower):
        base = now.replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    # "tomorrow"
    if re.search(r"\btomorrow\b", lower):
        base = (now + timedelta(days=1)).replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    # "this <weekday>" or bare weekday
    m = _THIS_WEEKDAY_RE.search(lower)
    if m:
        base = _next_occurrence(m.group(2))
        return _apply_time(base, lower).isoformat()

    # standalone time only → assume today
    if _TIME_RE.search(lower):
        base = now.replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    return None


def is_ambiguous(text: str) -> bool:
    """True when the text contains no recognisable deterministic time token."""
    return extract_datetime(text) is None
