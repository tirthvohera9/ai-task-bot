"""
Regex-based intent detection — handles common, unambiguous phrasings only.
Anything uncertain is passed to the AI (OpenRouter) for deeper understanding.
"""
import re
from dataclasses import dataclass
from typing import Optional

from utils.datetime_parser import extract_datetime


@dataclass
class ParseResult:
    action: str                        # add | list | delete | done
    title: str = ""
    datetime_iso: Optional[str] = None
    priority: str = "medium"
    confidence: float = 1.0
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Action patterns
# ---------------------------------------------------------------------------

# ADD — explicit "add/create/remind/schedule" AND implicit "I need to / don't forget"
_ADD = re.compile(
    r"^(add|create|new|schedule|set\s+(a\s+)?(reminder|task|alarm)|"
    r"remind(\s+me(\s+to)?)?|note(\s+down)?|book|"
    r"i\s+need\s+to\s+(?!check|see|know|find|look|ask)|"  # "I need to X" (exclude queries)
    r"i\s+have\s+to\s+(?!check|see|know|find|look|ask)|"
    r"don'?t\s+forget(\s+to)?|gotta\s+|"
    r"i\s+should\s+(?!check|see|know|find|look|ask)|"
    r"put(\s+it)?\s+on\s+(my\s+)?(list|calendar))\b",
    re.IGNORECASE,
)

# LIST — conservative: only explicit "list/show tasks" patterns
# Anything like "show milk" or "show doctor" should go to AI as a search
_LIST = re.compile(
    r"^(list|show|display|get)\s+(all\s+)?(my\s+)?(tasks?|todos?|reminders?|schedule|everything)\b"
    r"|"
    r"\b(today'?s?|tomorrow'?s?)\s+(tasks?|todos?|reminders?|schedule)\b"
    r"|"
    r"^what\s+(?:do\s+i\s+have\s+|is\s+on\s+my\s+)?(?:today|tomorrow)\??$"
    r"|"
    r"^(show|list)\s+(today|tomorrow)\b",
    re.IGNORECASE,
)

# DELETE — explicit removal verbs
_DELETE = re.compile(
    r"^(delete|remove|cancel|drop|clear|erase)\b",
    re.IGNORECASE,
)

# DONE — explicit completion + natural "I did / I finished / just completed"
_DONE = re.compile(
    r"^(done|finish(ed)?|complet(e|ed)|mark\s+(?:as\s+)?done|"
    r"i\s+(?:did|finished|completed|have\s+done|'?ve\s+(?:done|finished|completed))|"
    r"just\s+(?:finished|completed|done)|"
    r"(?:already\s+)?(?:finished|completed)\s+(?:the\s+)?)\b",
    re.IGNORECASE,
)

# Priority keywords
_PRIORITY_RE = re.compile(
    r"\b(urgent|critical|asap|high[\s-]?priority|important|low[\s-]?priority|low|minor)\b",
    re.IGNORECASE,
)
_PRIORITY_MAP = {
    "urgent": "high", "critical": "high", "asap": "high",
    "high priority": "high", "high-priority": "high", "important": "high",
    "low": "low", "minor": "low", "low priority": "low", "low-priority": "low",
}

# Tokens to strip when building a clean title
_NOISE_RE = re.compile(
    r"^(add|create|new|remind(\s+me(\s+to)?)?|schedule|set(\s+a)?\s*(reminder|task|alarm)?|"
    r"delete|remove|cancel|drop|clear|erase|"
    r"done|finish(ed)?|complete(d)?|mark(\s+as)?\s+done|"
    r"list|show|display|get|"
    r"i\s+(need|have)\s+to|don'?t\s+forget(\s+to)?|gotta|i\s+should|"
    r"put(\s+it)?\s+on\s+(my\s+)?(list|calendar)|"
    r"note(\s+down)?|book|i\s+(did|finished|completed)|i'?ve\s+(done|finished|completed)|"
    r"just\s+(finished|completed|done)|already\s+(finished|completed))\s*",
    re.IGNORECASE,
)
_TIME_TOKENS_RE = re.compile(
    r"\b(today|tomorrow|tonight|this\s+(morning|afternoon|evening|week|weekend)|"
    r"next\s+(week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"at\s+\d{1,2}(:\d{2})?\s*(am|pm)?|"
    r"in\s+\d+\s+(minute|hour|day|week|month)s?|"
    r"(this\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"end\s+of\s+(the\s+)?(week|month))\b",
    re.IGNORECASE,
)
_PRIORITY_TOKEN_RE = re.compile(
    r"\b(urgent|critical|asap|high[\s-]?priority|low[\s-]?priority|low|minor|important)\b",
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
    # strip trailing punctuation and collapse spaces
    title = re.sub(r"\s{2,}", " ", title).strip(" ,.")
    return title


def parse(text: str) -> Optional[ParseResult]:
    """
    Attempt deterministic parsing.
    Returns ParseResult on success, None if intent is ambiguous (→ escalate to AI).
    """
    stripped = text.strip()

    # --- LIST (conservative) ---
    if _LIST.search(stripped):
        return ParseResult(action="list", raw_text=stripped)

    # --- ADD ---
    if _ADD.match(stripped):
        title = _clean_title(stripped)
        if not title or len(title) < 2:
            return None  # no usable title — let AI handle it
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

    # Anything else: let AI figure it out
    return None
