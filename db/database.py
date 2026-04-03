"""
Redis layer via Upstash — replaces SQLite for Vercel serverless compatibility.
All data is stored in Upstash Redis (free tier: 10,000 commands/day).

Key schema:
  cache:{sha256}          → JSON string (parsed intent, TTL 24h)
  behavior:{user_id}      → Redis list of JSON action logs
  reminder:{page_id}      → "1" (TTL 24h, dedup sent reminders)
  config:{key}            → string value (e.g. notion_database_id)
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from upstash_redis.asyncio import Redis

from config import settings

logger = logging.getLogger(__name__)

# Single shared Redis client (connections are stateless HTTP REST calls)
_redis = Redis(
    url=settings.UPSTASH_REDIS_REST_URL,
    token=settings.UPSTASH_REDIS_REST_TOKEN,
)

CACHE_TTL = 60 * 60 * 24        # 24 hours
REMINDER_TTL = 60 * 60 * 24     # 24 hours (prevents re-sending same reminder)


# ---------------------------------------------------------------------------
# Input cache
# ---------------------------------------------------------------------------
def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


async def get_cached(text: str) -> Optional[dict]:
    raw = await _redis.get(f"cache:{_hash(text)}")
    if raw is None:
        return None
    return json.loads(raw)


async def set_cache(text: str, result: dict) -> None:
    await _redis.set(
        f"cache:{_hash(text)}",
        json.dumps(result),
        ex=CACHE_TTL,
    )


# ---------------------------------------------------------------------------
# User behaviour
# ---------------------------------------------------------------------------
async def log_behavior(user_id: str, action: str, task_title: str = "") -> None:
    entry = json.dumps({
        "action": action,
        "task_title": task_title,
        "occurred_at": _now(),
    })
    await _redis.lpush(f"behavior:{user_id}", entry)
    # Keep only the last 100 entries per user
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
# Sent-reminder dedup
# ---------------------------------------------------------------------------
async def is_reminder_sent(notion_page_id: str) -> bool:
    result = await _redis.exists(f"reminder:{notion_page_id}")
    return result > 0


async def mark_reminder_sent(notion_page_id: str) -> None:
    await _redis.set(f"reminder:{notion_page_id}", "1", ex=REMINDER_TTL)


# ---------------------------------------------------------------------------
# App config  (e.g. notion_database_id persisted across cold starts)
# ---------------------------------------------------------------------------
async def get_config(key: str) -> Optional[str]:
    value = await _redis.get(f"config:{key}")
    return value


async def set_config(key: str, value: str) -> None:
    await _redis.set(f"config:{key}", value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
