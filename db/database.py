"""
Redis layer via Upstash — replaces SQLite for Vercel serverless compatibility.

Key schema:
  cache:{sha256}          → JSON intent (TTL 24h)
  behavior:{user_id}      → list of JSON action logs (max 100)
  reminder:{page_id}      → "1" (TTL per priority, dedup sent reminders)
  config:{key}            → string value (e.g. notion_database_id)
  history:{user_id}       → list of JSON {role, content} messages (TTL 2h)
  tz:{user_id}            → IANA timezone name (e.g. "Asia/Kolkata")
  pending:{user_id}       → JSON pending task awaiting confirmation (TTL 5min)
  pattern:{user_id}       → JSON behavioral pattern data
  chat_streak:{user_id}   → int, consecutive chat-only turns (TTL 2h, reset on task)
  memo:{user_id}          → JSON {key: value} long-term user signals (no TTL)
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from upstash_redis.asyncio import Redis

from config import settings

logger = logging.getLogger(__name__)

_redis = Redis(
    url=settings.KV_REST_API_URL,
    token=settings.KV_REST_API_TOKEN,
)

CACHE_TTL    = 60 * 60 * 24      # 24h
REMINDER_TTL = 60 * 60 * 24      # 24h (high priority gets multiple sends)
HISTORY_TTL  = 60 * 60 * 2       # 2h inactivity resets context
PENDING_TTL  = 60 * 5            # 5min to confirm a pending task
MAX_HISTORY  = 10


# ---------------------------------------------------------------------------
# Intent cache
# ---------------------------------------------------------------------------
def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


async def get_cached(text: str) -> Optional[dict]:
    raw = await _redis.get(f"cache:{_hash(text)}")
    return json.loads(raw) if raw else None


async def set_cache(text: str, result: dict) -> None:
    await _redis.set(f"cache:{_hash(text)}", json.dumps(result), ex=CACHE_TTL)


# ---------------------------------------------------------------------------
# User timezone  (IANA name, e.g. "Asia/Kolkata", "America/New_York")
# ---------------------------------------------------------------------------
async def set_user_timezone(user_id: str, tz_name: str) -> None:
    await _redis.set(f"tz:{user_id}", tz_name)


async def get_user_timezone(user_id: str) -> str:
    tz = await _redis.get(f"tz:{user_id}")
    return tz or "Asia/Kolkata"


# ---------------------------------------------------------------------------
# Pending task confirmation  (multi-turn yes/no flow)
# ---------------------------------------------------------------------------
async def set_pending_task(user_id: str, task: dict) -> None:
    """Store an unconfirmed task for up to 5 minutes."""
    await _redis.set(f"pending:{user_id}", json.dumps(task), ex=PENDING_TTL)


async def get_pending_task(user_id: str) -> Optional[dict]:
    raw = await _redis.get(f"pending:{user_id}")
    return json.loads(raw) if raw else None


async def clear_pending_task(user_id: str) -> None:
    await _redis.delete(f"pending:{user_id}")


# ---------------------------------------------------------------------------
# Behavioral patterns  (category frequency, preferred hour per category)
# ---------------------------------------------------------------------------
async def record_behavior_pattern(
    user_id: str,
    category: Optional[str],
    hour: Optional[int],
) -> None:
    """Track which categories and hours the user most often uses."""
    key = f"pattern:{user_id}"
    raw = await _redis.get(key)
    patterns: dict = json.loads(raw) if raw else {}

    if category:
        cat_counts = patterns.get("categories", {})
        cat_counts[category] = cat_counts.get(category, 0) + 1
        patterns["categories"] = cat_counts

    if hour is not None and category:
        hour_key = f"hours_{category}"
        hours = patterns.get(hour_key, {})
        hours[str(hour)] = hours.get(str(hour), 0) + 1
        patterns[hour_key] = hours

    await _redis.set(key, json.dumps(patterns))


async def get_behavior_patterns(user_id: str) -> dict:
    raw = await _redis.get(f"pattern:{user_id}")
    return json.loads(raw) if raw else {}


# ---------------------------------------------------------------------------
# User behaviour log
# ---------------------------------------------------------------------------
async def log_behavior(user_id: str, action: str, task_title: str = "") -> None:
    entry = json.dumps({
        "action": action,
        "task_title": task_title,
        "occurred_at": _now(),
    })
    await _redis.lpush(f"behavior:{user_id}", entry)
    await _redis.ltrim(f"behavior:{user_id}", 0, 99)


async def get_behavior_summary(user_id: str) -> dict:
    entries = await _redis.lrange(f"behavior:{user_id}", 0, -1)
    summary: dict = {}
    for raw in entries:
        data = json.loads(raw)
        action = data.get("action", "unknown")
        summary[action] = summary.get(action, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# Sent-reminder dedup  (TTL varies by priority)
# ---------------------------------------------------------------------------
async def is_reminder_sent(notion_page_id: str, slot: str = "main") -> bool:
    result = await _redis.exists(f"reminder:{notion_page_id}:{slot}")
    return result > 0


async def mark_reminder_sent(
    notion_page_id: str,
    slot: str = "main",
    ttl: int = REMINDER_TTL,
) -> None:
    await _redis.set(f"reminder:{notion_page_id}:{slot}", "1", ex=ttl)


# ---------------------------------------------------------------------------
# Conversation history  (short-term context per user)
# ---------------------------------------------------------------------------
async def add_to_history(user_id: str, role: str, content: str) -> None:
    entry = json.dumps({"role": role, "content": content})
    key = f"history:{user_id}"
    await _redis.lpush(key, entry)
    await _redis.ltrim(key, 0, MAX_HISTORY - 1)
    await _redis.expire(key, HISTORY_TTL)


async def get_history(user_id: str) -> list[dict]:
    entries = await _redis.lrange(f"history:{user_id}", 0, -1)
    return [json.loads(e) for e in reversed(entries)] if entries else []


async def clear_history(user_id: str) -> None:
    await _redis.delete(f"history:{user_id}")


# ---------------------------------------------------------------------------
# Chat streak  (consecutive non-task turns — drift detection)
# ---------------------------------------------------------------------------
CHAT_STREAK_TTL = 60 * 60 * 2  # 2h — resets with inactivity


async def increment_chat_streak(user_id: str) -> int:
    """Increment and return the current chat streak count."""
    key = f"chat_streak:{user_id}"
    count = await _redis.incr(key)
    if count == 1:
        await _redis.expire(key, CHAT_STREAK_TTL)
    return int(count)


async def get_chat_streak(user_id: str) -> int:
    val = await _redis.get(f"chat_streak:{user_id}")
    return int(val) if val else 0


async def reset_chat_streak(user_id: str) -> None:
    await _redis.delete(f"chat_streak:{user_id}")


# ---------------------------------------------------------------------------
# User memos  (long-term behavioral signals, no TTL)
# ---------------------------------------------------------------------------
async def set_user_memo(user_id: str, key: str, value: str) -> None:
    """Store a long-term signal about the user (e.g. 'prefers_evening': 'true')."""
    memo_key = f"memo:{user_id}"
    raw = await _redis.get(memo_key)
    memos: dict = json.loads(raw) if raw else {}
    memos[key] = value
    await _redis.set(memo_key, json.dumps(memos))


async def get_user_memos(user_id: str) -> dict:
    raw = await _redis.get(f"memo:{user_id}")
    return json.loads(raw) if raw else {}


# ---------------------------------------------------------------------------
# App config  (persists across cold starts)
# ---------------------------------------------------------------------------
async def get_config(key: str) -> Optional[str]:
    return await _redis.get(f"config:{key}")


async def set_config(key: str, value: str) -> None:
    await _redis.set(f"config:{key}", value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
