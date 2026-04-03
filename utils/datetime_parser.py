"""
Deterministic datetime extraction from natural-language strings.
Returns an ISO-8601 string or None.  No AI involved.

Handles: today, tomorrow, tonight, this morning/afternoon/evening,
         this/next weekday, next week, this weekend, end of week/month,
         "April 5th", "5th April", "the 15th", in X minutes/hours/days/weeks/months
"""
import calendar
import re
from datetime import datetime, timedelta, timezone
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ── Time patterns ─────────────────────────────────────────────────────────────
_TIME_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)
_RELATIVE_RE = re.compile(
    r"\bin\s+(\d+)\s+(minute|hour|day|week|month)s?\b",
    re.IGNORECASE,
)
_NEXT_WEEKDAY_RE = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)\b",
    re.IGNORECASE,
)
_THIS_WEEKDAY_RE = re.compile(
    r"\b(?:this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)\b",
    re.IGNORECASE,
)
# "April 5" / "April 5th"
_MONTH_DAY_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
# "5th April" / "5 April"
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august"
    r"|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b",
    re.IGNORECASE,
)
# "the 5th"
_ORDINAL_DAY_RE = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


def _apply_time(base: datetime, text: str) -> datetime:
    """Overlay HH:MM from text onto base date; if no time token defaults to 09:00."""
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
    # Heuristic: bare "at 5" / "at 6" → 5pm/6pm (people rarely mean 5am)
    elif not meridiem and 1 <= hour <= 6:
        hour += 12

    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _next_occurrence(weekday_name: str, skip_current: bool = False) -> datetime:
    """Return the next datetime whose weekday matches weekday_name."""
    target = _WEEKDAY_MAP[weekday_name.lower()]
    now = _now()
    days_ahead = target - now.weekday()
    if days_ahead < 0 or (days_ahead == 0 and skip_current):
        days_ahead += 7
    return now + timedelta(days=days_ahead)


def extract_datetime(text: str) -> Optional[str]:
    """
    Returns ISO-8601 UTC string when a date/time reference is found, else None.
    """
    lower = text.lower()
    now = _now()

    # ── "in X minutes / hours / days / weeks / months" ───────────────────────
    m = _RELATIVE_RE.search(lower)
    if m:
        amount, unit = int(m.group(1)), m.group(2)
        deltas = {
            "minute": timedelta(minutes=amount),
            "hour":   timedelta(hours=amount),
            "day":    timedelta(days=amount),
            "week":   timedelta(weeks=amount),
            "month":  timedelta(days=amount * 30),
        }
        return (now + deltas[unit]).isoformat()

    # ── Time-of-day keywords ──────────────────────────────────────────────────
    if re.search(r"\btonight\b", lower):
        base = now.replace(second=0, microsecond=0)
        return (_apply_time(base, lower) if _TIME_RE.search(lower)
                else base.replace(hour=20, minute=0, second=0, microsecond=0)).isoformat()

    if re.search(r"\bthis\s+evening\b", lower):
        base = now.replace(second=0, microsecond=0)
        return (_apply_time(base, lower) if _TIME_RE.search(lower)
                else base.replace(hour=19, minute=0, second=0, microsecond=0)).isoformat()

    if re.search(r"\bthis\s+morning\b", lower):
        base = now.replace(second=0, microsecond=0)
        return (_apply_time(base, lower) if _TIME_RE.search(lower)
                else base.replace(hour=9, minute=0, second=0, microsecond=0)).isoformat()

    if re.search(r"\bthis\s+afternoon\b", lower):
        base = now.replace(second=0, microsecond=0)
        return (_apply_time(base, lower) if _TIME_RE.search(lower)
                else base.replace(hour=14, minute=0, second=0, microsecond=0)).isoformat()

    # ── "next week" → next Monday ─────────────────────────────────────────────
    if re.search(r"\bnext\s+week\b", lower):
        days_ahead = (7 - now.weekday()) % 7 or 7
        base = (now + timedelta(days=days_ahead)).replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    # ── "this weekend" / "weekend" → coming Saturday ─────────────────────────
    if re.search(r"\bthis\s+weekend\b|\bweekend\b", lower):
        days_ahead = (5 - now.weekday()) % 7 or 7
        base = (now + timedelta(days=days_ahead)).replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    # ── "end of (the) month" → last day of current month ─────────────────────
    if re.search(r"\bend\s+of\s+(?:the\s+)?month\b", lower):
        last_day = calendar.monthrange(now.year, now.month)[1]
        base = now.replace(day=last_day, hour=9, minute=0, second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    # ── "end of (the) week" → coming Friday ──────────────────────────────────
    if re.search(r"\bend\s+of\s+(?:the\s+)?week\b", lower):
        days_ahead = (4 - now.weekday()) % 7 or 7
        base = (now + timedelta(days=days_ahead)).replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    # ── "next <weekday>" ──────────────────────────────────────────────────────
    m = _NEXT_WEEKDAY_RE.search(lower)
    if m:
        base = _next_occurrence(m.group(1), skip_current=True)
        return _apply_time(base, lower).isoformat()

    # ── "today" / "tomorrow" ─────────────────────────────────────────────────
    if re.search(r"\btoday\b", lower):
        base = now.replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    if re.search(r"\btomorrow\b", lower):
        base = (now + timedelta(days=1)).replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    # ── "April 5" / "April 5th" ───────────────────────────────────────────────
    m = _MONTH_DAY_RE.search(lower)
    if m:
        month_num = _MONTH_MAP.get(m.group(1).lower())
        day = int(m.group(2))
        if month_num and 1 <= day <= 31:
            try:
                base = now.replace(month=month_num, day=day, second=0, microsecond=0)
                if base < now:
                    base = base.replace(year=now.year + 1)
                return _apply_time(base, lower).isoformat()
            except ValueError:
                pass

    # ── "5th April" ──────────────────────────────────────────────────────────
    m = _DAY_MONTH_RE.search(lower)
    if m:
        day = int(m.group(1))
        month_num = _MONTH_MAP.get(m.group(2).lower())
        if month_num and 1 <= day <= 31:
            try:
                base = now.replace(month=month_num, day=day, second=0, microsecond=0)
                if base < now:
                    base = base.replace(year=now.year + 1)
                return _apply_time(base, lower).isoformat()
            except ValueError:
                pass

    # ── "the 5th" → 5th of this/next month ───────────────────────────────────
    m = _ORDINAL_DAY_RE.search(lower)
    if m:
        day = int(m.group(1))
        try:
            base = now.replace(day=day, second=0, microsecond=0)
            if base < now:
                next_month = now.month % 12 + 1
                year = now.year + (1 if now.month == 12 else 0)
                base = base.replace(year=year, month=next_month)
            return _apply_time(base, lower).isoformat()
        except ValueError:
            pass

    # ── "this <weekday>" or bare weekday ─────────────────────────────────────
    m = _THIS_WEEKDAY_RE.search(lower)
    if m:
        base = _next_occurrence(m.group(1))
        return _apply_time(base, lower).isoformat()

    # ── Standalone time only → assume today ──────────────────────────────────
    if _TIME_RE.search(lower):
        base = now.replace(second=0, microsecond=0)
        return _apply_time(base, lower).isoformat()

    return None


def is_ambiguous(text: str) -> bool:
    """True when the text contains no recognisable deterministic time token."""
    return extract_datetime(text) is None
