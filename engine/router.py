"""
Command router: regex-first, Claude fallback.

Decision flow:
  1. Check SQLite cache → return cached result if hit
  2. Try deterministic regex parse
  3. If regex succeeds → return ParseResult (no AI cost)
  4. If regex fails  → call Claude, cache the result
"""
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from db.database import get_cached, set_cache
from services.ai_service import parse_intent
from utils.regex_parser import ParseResult, parse

logger = logging.getLogger(__name__)


import re as _re

# Inputs that are context-dependent must never be cached — their meaning
# changes based on what was shown last (pronoun or number reference).
_NO_CACHE_RE = _re.compile(
    r"^\s*(?:"
    r"\d+"                                          # bare number: "1", "2"
    r"|(?:delete|done|remove|complete|move|reschedule|update|mark)\s+"
    r"(?:it|that|this|one|\d+)\b"                  # "delete it", "done 2"
    r"|(?:it|that|this)\s+(?:is|was|done|complete)" # "that is done"
    r")\s*$",
    _re.IGNORECASE,
)


async def route(text: str, history: Optional[list] = None) -> Optional[dict]:
    """
    Returns a dict matching the intent schema, or None if unresolvable.

    Schema:
    {
        "action":     "add|list|delete|done|update",
        "title":      str,
        "datetime":   ISO-8601 str | None,
        "priority":   "low|medium|high",
        "entity":     str | None,
        "confidence": float,
        "source":     "regex|claude|cache"
    }
    """
    normalised = text.strip()

    # Context-dependent inputs must bypass cache entirely
    cacheable = not _NO_CACHE_RE.match(normalised)

    # ── 1. Cache lookup ───────────────────────────────────────────────────
    if cacheable:
        cached = await get_cached(normalised)
        if cached:
            logger.debug("Cache hit for: %s", normalised)
            cached["source"] = "cache"
            return cached

    # ── 2. Deterministic regex parse ──────────────────────────────────────
    result: Optional[ParseResult] = parse(normalised)
    if result is not None:
        intent = {
            **asdict(result),
            "datetime": result.datetime_iso,
            "entity": None,
            "source": "regex",
        }
        intent.pop("datetime_iso", None)
        intent.pop("raw_text", None)
        logger.info("Regex parsed action=%s title=%s", intent["action"], intent["title"])
        if cacheable:
            await set_cache(normalised, intent)
        return intent

    # ── 3. Claude fallback ────────────────────────────────────────────────
    logger.info("Escalating to Claude for: %s", normalised)
    now = datetime.now(timezone.utc).isoformat()
    claude_result = await parse_intent(normalised, current_time=now, history=history)

    if claude_result is None:
        logger.warning("Claude could not parse: %s", normalised)
        return None

    claude_result["source"] = "claude"
    if cacheable:
        await set_cache(normalised, claude_result)
    return claude_result
