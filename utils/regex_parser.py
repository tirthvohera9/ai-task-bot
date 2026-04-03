"""
Regex-based intent detection.
Returns a ParseResult if the input is unambiguous, otherwise None
(caller must escalate to Claude).
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from utils.datetime_parser import extract_datetime


@dataclass
class ParseResult:
    action: str                       # add | list | delete | done | update
    title: str = ""
    datetime_iso: Optional[str] = None
    priority: str = "medium"          # low | medium | high
    confidence: float = 1.0
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Action keyword patterns
# ---------------------------------------------------------------------------
_ADD = re.compile(
    r"^(add|create|new|remind|schedule|set)\b",
    re.IGNORECASE,
)
_LIST = re.compile(
    r"^(list|show|get|what('?s| is)|display|tell me)\b"
    r"|"
    r"\b(today'?s?|tomorrow'?s?)\s+(tasks?|todos?|reminders?)\b",
    re.IGNORECASE,
)
_DELETE = re.compile(
    r"^(delete|remove|cancel|drop)\b",
    re.IGNORECASE,
)
_DONE = re.compile(
    r"^(done|finish(ed)?|complet(e|ed)|mark\s+(as\s+)?done)\b",
    re.IGNORECASE,
)

# Priority keywords inside the message
_PRIORITY_RE = re.compile(
    r"\b(urgent|critical|asap|high|important|low|minor|low[- ]priority)\b",
    re.IGNORECASE,
)
_PRIORITY_MAP = {
    "urgent": "high", "critical": "high", "asap": "high",
    "high": "high", "important": "high",
    "low": "low", "minor": "low", "low-priority": "low", "low priority": "low",
}

# Tokens to strip when extracting a clean title
_NOISE_RE = re.compile(
    r"^(add|create|new|remind\s+(me\s+)?to|schedule|set|delete|remove|"
    r"cancel|done|finish|complete|mark\s+as\s+done|list|show|get|display)\s*",
    re.IGNORECASE,
)
_TIME_TOKENS_RE = re.compile(
    r"\b(today|tomorrow|at\s+\d{1,2}(:\d{2})?\s*(am|pm)?|"
    r"in\s+\d+\s+(minute|hour|day)s?|"
    r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
_PRIORITY_TOKEN_RE = re.compile(
    r"\b(urgent|critical|asap|high\s+priority|low\s+priority|low|minor|important)\b",
    re.IGNORECASE,
)


def _extract_priority(text: str) -> str:
    m = _PRIORITY_RE.search(text)
    if not m:
        return "medium"
    token = m.group(1).lower().replace("-", " ")
    return _PRIORITY_MAP.get(token, "medium")


def _clean_title(text: str) -> str:
    title = _NOISE_RE.sub("", text).strip()
    title = _TIME_TOKENS_RE.sub("", title).strip()
    title = _PRIORITY_TOKEN_RE.sub("", title).strip()
    # Collapse multiple spaces
    return re.sub(r"\s{2,}", " ", title).strip(" ,.")


def parse(text: str) -> Optional[ParseResult]:
    """
    Attempt deterministic parsing.
    Returns ParseResult on success, None if intent is ambiguous.
    """
    stripped = text.strip()

    # --- LIST ---
    if _LIST.search(stripped):
        # No datetime needed for list commands
        return ParseResult(action="list", raw_text=stripped)

    # --- ADD ---
    if _ADD.match(stripped):
        title = _clean_title(stripped)
        if not title:
            return None  # title required; escalate to Claude
        dt = extract_datetime(stripped)
        priority = _extract_priority(stripped)
        return ParseResult(
            action="add",
            title=title,
            datetime_iso=dt,
            priority=priority,
            raw_text=stripped,
        )

    # --- DELETE ---
    if _DELETE.match(stripped):
        title = _clean_title(stripped)
        if not title:
            return None
        return ParseResult(action="delete", title=title, raw_text=stripped)

    # --- DONE ---
    if _DONE.match(stripped):
        title = _clean_title(stripped)
        if not title:
            return None
        return ParseResult(action="done", title=title, raw_text=stripped)

    # Cannot determine intent deterministically
    return None
