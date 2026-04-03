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
You are a task-extraction engine with long-term memory via Notion.
Given a user message, return ONLY a single valid JSON object — no explanation, no markdown.

Schema:
{
  "action": "add|list|delete|done|search",
  "title": "task title (for add/delete/done) or null",
  "keyword": "keyword to search tasks by (for search/list queries) or null",
  "datetime": "ISO8601 UTC string or null",
  "date_range": "today|tomorrow|this_week|last_7_days|last_15_days|last_30_days|all or null",
  "priority": "low|medium|high",
  "include_done": false,
  "confidence": 0.0
}

Rules:
- action=search: user asks about specific tasks ("when do I buy milk?", "what did I plan for Friday?")
- action=list: user wants all tasks in a time window ("reminders for tomorrow", "last 15 days")
- keyword: extract the subject ("milk" from "when do I have to buy milk")
- date_range: extract from "last 15 days", "tomorrow", "this week", "last month" etc.
- include_done=true when user asks about past/historical tasks
- Return ONLY the JSON. No other text."""


async def parse_intent(
    text: str,
    current_time: Optional[str] = None,
    history: Optional[list] = None,
) -> Optional[dict]:
    """
    Send text to OpenRouter and return a parsed intent dict.
    history: recent conversation turns for context (e.g. resolving "5pm" follow-ups).
    Returns None on any failure.
    """
    now = current_time or datetime.now(timezone.utc).isoformat()
    user_message = f"Current UTC time: {now}\nUser input: {text}"

    # Build messages: system + recent history (last 4 msgs) + current input
    messages: list = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_message})

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=200,
            temperature=0,
            messages=messages,
        )
        raw = response.choices[0].message.content.strip()
        logger.debug("OpenRouter raw response: %s", raw)
        return _parse_json(raw)

    except Exception as exc:
        logger.error("OpenRouter parse_intent failed: %s", exc)
        return None


async def general_chat(text: str, history: Optional[list] = None) -> str:
    """
    Conversational fallback for non-task messages.
    Includes conversation history so follow-ups like "5pm" are understood.
    """
    _CHAT_SYSTEM = (
        "You are a helpful AI assistant and personal task manager. "
        "Answer questions naturally and concisely. "
        "Remember prior messages in this conversation for context. "
        "If the user wants to manage tasks, remind them: "
        "'Add meeting at 3pm' or 'Show today's tasks'."
    )

    messages: list = [{"role": "system", "content": _CHAT_SYSTEM}]
    if history:
        messages.extend(history[-6:])   # last 6 messages for context
    messages.append({"role": "user", "content": text})

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            max_tokens=300,
            temperature=0.7,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("OpenRouter general_chat failed: %s", exc)
        return "Sorry, I couldn't process that right now. Please try again."


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
