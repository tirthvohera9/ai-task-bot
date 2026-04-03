"""
SQLite layer: input cache, user behaviour tracking, sent-reminder dedup,
and storage for the Notion database ID.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

DB_PATH = "tasks.db"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS input_cache (
    hash        TEXT PRIMARY KEY,
    input_text  TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_behavior (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    action       TEXT NOT NULL,        -- 'created'|'completed'|'deleted'|'delayed'
    task_title   TEXT,
    occurred_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sent_reminders (
    notion_page_id TEXT PRIMARY KEY,
    sent_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES)
        await db.commit()
    logger.info("SQLite initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Input cache
# ---------------------------------------------------------------------------
def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


async def get_cached(text: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT result_json FROM input_cache WHERE hash = ?", (_hash(text),)
        ) as cur:
            row = await cur.fetchone()
    return json.loads(row[0]) if row else None


async def set_cache(text: str, result: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO input_cache (hash, input_text, result_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (_hash(text), text, json.dumps(result), _now()),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# User behaviour
# ---------------------------------------------------------------------------
async def log_behavior(user_id: str, action: str, task_title: str = "") -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO user_behavior (user_id, action, task_title, occurred_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, action, task_title, _now()),
        )
        await db.commit()


async def get_behavior_summary(user_id: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT action, COUNT(*) FROM user_behavior WHERE user_id = ? GROUP BY action",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return {row[0]: row[1] for row in rows}


# ---------------------------------------------------------------------------
# Sent-reminder dedup
# ---------------------------------------------------------------------------
async def is_reminder_sent(notion_page_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT 1 FROM sent_reminders WHERE notion_page_id = ?", (notion_page_id,)
        ) as cur:
            return await cur.fetchone() is not None


async def mark_reminder_sent(notion_page_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sent_reminders (notion_page_id, sent_at) VALUES (?, ?)",
            (notion_page_id, _now()),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# App config (stores notion_database_id, etc.)
# ---------------------------------------------------------------------------
async def get_config(key: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT value FROM app_config WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


async def set_config(key: str, value: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO app_config (key, value) VALUES (?, ?)", (key, value)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
