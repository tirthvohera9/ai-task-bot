"""
OpenRouter AI service — the single brain for all AI decisions.

Uses the OpenAI-compatible API endpoint provided by OpenRouter, so any
model available on openrouter.ai can be swapped in via OPENROUTER_MODEL.

Called ONLY when deterministic regex routing fails.
Max tokens: 200  |  No conversation history sent.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

# OpenRouter exposes an OpenAI-compatible endpoint
_client = AsyncOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/ai-task-manager",  # optional but recommended
        "X-Title": "AI Task Manager",
    },
)

_SYSTEM_PROMPT = """\
You are a task-extraction engine.
Given a user message, return ONLY a single valid JSON object — no explanation, no markdown.

Schema:
{"action":"add|list|delete|done|update","title":"string","datetime":"ISO8601 UTC or null","priority":"low|medium|high","entity":"string or null","confidence":0.0}

Rules:
- action must be exactly one of: add, list, delete, done, update
- datetime must be a full ISO-8601 UTC string (e.g. 2025-06-01T17:00:00+00:00) or null
- confidence: 0.0–1.0, how certain you are
- Return ONLY the JSON object. No other text."""


async def parse_intent(text: str, current_time: Optional[str] = None) -> Optional[dict]:
    """
    Send text to OpenRouter and return a parsed intent dict.
    Returns None on any failure.
    """
    now = current_time or datetime.now(timezone.utc).isoformat()
    user_message = f"Current UTC time: {now}\nUser input: {text}"

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=200,
            temperature=0,          # deterministic output
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        raw = response.choices[0].message.content.strip()
        logger.debug("OpenRouter raw response: %s", raw)
        return _parse_json(raw)

    except Exception as exc:
        logger.error("OpenRouter parse_intent failed: %s", exc)
        return None


def _parse_json(raw: str) -> Optional[dict]:
    # Strip accidental markdown fences
    raw = raw.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("OpenRouter returned non-JSON: %s", raw)
        return None

    required = {"action", "title", "datetime", "priority", "confidence"}
    if not required.issubset(data.keys()):
        logger.warning("OpenRouter JSON missing keys: %s", data)
        return None

    if data["action"] not in {"add", "list", "delete", "done", "update"}:
        logger.warning("OpenRouter returned unknown action: %s", data["action"])
        return None

    return data
