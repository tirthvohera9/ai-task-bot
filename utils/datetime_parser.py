"""
Deterministic datetime extraction from natural-language strings.
Returns an ISO-8601 UTC string or None. No AI involved.

Timezone-aware: pass user_tz (IANA name) so "at 5pm" = 5pm in the user's
local timezone, stored as UTC in Notion.

Supported expressions:
  in X minutes/hours/days/weeks/months
  today, tomorrow, tonight, this morning/afternoon/evening
  this/next weekday, next week, this weekend
  end of week/month
  April 5th, 5th April, the 15th
  at HH:MM am/pm
"""
import calendar
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _tz(tz_name: str) -> timezone:
    """Return a timezone object; fall back to UTC on invalid names."""
    try:
        return ZoneInfo(tz_name)  # type: ignore[return-value]
    except (ZoneInfoNotFoundError, Exception):
        return timezone.utc


def _now(tz_name: str = "UTC") -> datetime:
    return datetime.now(_tz(tz_name))


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2,
    "thu": 3, "thur": 3, "thurs": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"  # "at 5", "at 5pm", "at 5:30 am"
    r"|\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",     # bare "10am", "10 am", "10:30am" (am/pm required)
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
_MONTH_DAY_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october"
    r"|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(january|february|march|april|may|june|july|august|september|october"
    r"|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b",
    re.IGNORECASE,
)
_ORDINAL_DAY_RE = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_time(base: datetime, text: str) -> datetime:
    """Overlay HH:MM from text onto base; default 09:00 if no time found.

    Handles two forms:
      Alt 1 — "at 5", "at 5pm", "at 5:30"   → groups 1,2,3
      Alt 2 — "10am", "10 am", "10:30am"     → groups 4,5,6  (explicit am/pm required)
    """
    m = _TIME_RE.search(text)
    if not m:
        return base.replace(hour=9, minute=0, second=0, microsecond=0)

    if m.group(1) is not None:          # Alt 1: "at HH:MM am/pm"
        hour   = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        merid  = (m.group(3) or "").lower()
    else:                               # Alt 2: bare "HHam/pm"
        hour   = int(m.group(4))
        minute = int(m.group(5)) if m.group(5) else 0
        merid  = (m.group(6) or "").lower()

    if merid == "pm" and hour < 12:
        hour += 12
    elif merid == "am" and hour == 12:
        hour = 0
    elif not merid and 1 <= hour <= 6:
        hour += 12   # heuristic: "at 5" → 5pm

    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _next_occurrence(weekday_name: str, now: datetime, skip_current: bool = False) -> datetime:
    target     = _WEEKDAY_MAP[weekday_name.lower()]
    days_ahead = target - now.weekday()
    if days_ahead < 0 or (days_ahead == 0 and skip_current):
        days_ahead += 7
    return now + timedelta(days=days_ahead)


def _to_utc_iso(dt: datetime) -> str:
    """Convert a timezone-aware datetime to UTC ISO-8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def extract_datetime(text: str, user_tz: str = "UTC") -> Optional[str]:
    """
    Returns ISO-8601 UTC string when a date/time reference is found, else None.
    All parsing is done in user_tz local time, then converted to UTC for storage.
    """
    lower = text.lower()
    now   = _now(user_tz)

    # ── "in X minutes/hours/days/weeks/months" ────────────────────────────
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
        return _to_utc_iso(now + deltas[unit])

    # ── Time-of-day keywords ──────────────────────────────────────────────
    base = now.replace(second=0, microsecond=0)

    if re.search(r"\btonight\b", lower):
        result = _apply_time(base, lower) if _TIME_RE.search(lower) \
                 else base.replace(hour=20, minute=0)
        return _to_utc_iso(result)

    if re.search(r"\bthis\s+evening\b", lower):
        result = _apply_time(base, lower) if _TIME_RE.search(lower) \
                 else base.replace(hour=19, minute=0)
        return _to_utc_iso(result)

    if re.search(r"\bthis\s+morning\b", lower):
        result = _apply_time(base, lower) if _TIME_RE.search(lower) \
                 else base.replace(hour=9, minute=0)
        return _to_utc_iso(result)

    if re.search(r"\bthis\s+afternoon\b", lower):
        result = _apply_time(base, lower) if _TIME_RE.search(lower) \
                 else base.replace(hour=14, minute=0)
        return _to_utc_iso(result)

    # ── "next week" → next Monday ─────────────────────────────────────────
    if re.search(r"\bnext\s+week\b", lower):
        days_ahead = (7 - now.weekday()) % 7 or 7
        b = (now + timedelta(days=days_ahead)).replace(second=0, microsecond=0)
        return _to_utc_iso(_apply_time(b, lower))

    # ── "this weekend" → coming Saturday ─────────────────────────────────
    if re.search(r"\bthis\s+weekend\b|\bweekend\b", lower):
        days_ahead = (5 - now.weekday()) % 7 or 7
        b = (now + timedelta(days=days_ahead)).replace(second=0, microsecond=0)
        return _to_utc_iso(_apply_time(b, lower))

    # ── "end of month" → last day of month ───────────────────────────────
    if re.search(r"\bend\s+of\s+(?:the\s+)?month\b", lower):
        last_day = calendar.monthrange(now.year, now.month)[1]
        b = now.replace(day=last_day, hour=9, minute=0, second=0, microsecond=0)
        return _to_utc_iso(_apply_time(b, lower))

    # ── "end of week" → coming Friday ─────────────────────────────────────
    if re.search(r"\bend\s+of\s+(?:the\s+)?week\b", lower):
        days_ahead = (4 - now.weekday()) % 7 or 7
        b = (now + timedelta(days=days_ahead)).replace(second=0, microsecond=0)
        return _to_utc_iso(_apply_time(b, lower))

    # ── "next <weekday>" ──────────────────────────────────────────────────
    m = _NEXT_WEEKDAY_RE.search(lower)
    if m:
        b = _next_occurrence(m.group(1), now, skip_current=True)
        return _to_utc_iso(_apply_time(b.replace(second=0, microsecond=0), lower))

    # ── "today" / "tomorrow" ─────────────────────────────────────────────
    if re.search(r"\btoday\b", lower):
        return _to_utc_iso(_apply_time(base, lower))

    if re.search(r"\btomorrow\b", lower):
        b = (now + timedelta(days=1)).replace(second=0, microsecond=0)
        return _to_utc_iso(_apply_time(b, lower))

    # ── "April 5" / "April 5th" ───────────────────────────────────────────
    m = _MONTH_DAY_RE.search(lower)
    if m:
        month_num = _MONTH_MAP.get(m.group(1).lower())
        day = int(m.group(2))
        if month_num and 1 <= day <= 31:
            try:
                b = now.replace(month=month_num, day=day, second=0, microsecond=0)
                if b < now:
                    b = b.replace(year=now.year + 1)
                return _to_utc_iso(_apply_time(b, lower))
            except ValueError:
                pass

    # ── "5th April" ──────────────────────────────────────────────────────
    m = _DAY_MONTH_RE.search(lower)
    if m:
        day = int(m.group(1))
        month_num = _MONTH_MAP.get(m.group(2).lower())
        if month_num and 1 <= day <= 31:
            try:
                b = now.replace(month=month_num, day=day, second=0, microsecond=0)
                if b < now:
                    b = b.replace(year=now.year + 1)
                return _to_utc_iso(_apply_time(b, lower))
            except ValueError:
                pass

    # ── "the 5th" → 5th of this/next month ───────────────────────────────
    m = _ORDINAL_DAY_RE.search(lower)
    if m:
        day = int(m.group(1))
        try:
            b = now.replace(day=day, second=0, microsecond=0)
            if b < now:
                next_month = now.month % 12 + 1
                yr = now.year + (1 if now.month == 12 else 0)
                b = b.replace(year=yr, month=next_month)
            return _to_utc_iso(_apply_time(b, lower))
        except ValueError:
            pass

    # ── "this <weekday>" or bare weekday ─────────────────────────────────
    m = _THIS_WEEKDAY_RE.search(lower)
    if m:
        b = _next_occurrence(m.group(1), now)
        return _to_utc_iso(_apply_time(b.replace(second=0, microsecond=0), lower))

    # ── Standalone time only → assume today ──────────────────────────────
    if _TIME_RE.search(lower):
        return _to_utc_iso(_apply_time(base, lower))

    return None


def is_ambiguous(text: str) -> bool:
    """True when the text contains no recognisable deterministic time token."""
    return extract_datetime(text) is None


def format_local(utc_iso: str, user_tz: str = "UTC") -> str:
    """
    Format a stored UTC ISO-8601 string as a human-readable local time string.
    e.g. "5 Apr at 5:00 PM IST"
    """
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(_tz(user_tz))

        # Get short timezone abbreviation
        tz_abbr = local_dt.strftime("%Z")
        return local_dt.strftime(f"%-d %b at %-I:%M %p {tz_abbr}")
    except Exception:
        return utc_iso
