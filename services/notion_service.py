"""
Notion integration: database bootstrap + task CRUD + smart queries.
Notion is the long-term memory — all tasks live here permanently.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from notion_client import AsyncClient

from config import settings
from db.database import get_config, set_config

logger = logging.getLogger(__name__)
_notion = AsyncClient(auth=settings.NOTION_API_KEY)

DB_CONFIG_KEY = "notion_database_id"

PROP_TITLE    = "Name"
PROP_DUE      = "Due"
PROP_PRIORITY = "Priority"
PROP_STATUS   = "Status"


# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------
async def get_or_create_database() -> str:
    db_id = await get_config(DB_CONFIG_KEY)
    if db_id:
        return db_id

    logger.info("Creating Notion tasks database…")
    response = await _notion.databases.create(
        parent={"type": "page_id", "page_id": settings.NOTION_PARENT_PAGE_ID},
        title=[{"type": "text", "text": {"content": "Tasks"}}],
        properties={
            PROP_TITLE: {"title": {}},
            PROP_DUE: {"date": {}},
            PROP_PRIORITY: {
                "select": {
                    "options": [
                        {"name": "high",   "color": "red"},
                        {"name": "medium", "color": "yellow"},
                        {"name": "low",    "color": "green"},
                    ]
                }
            },
            PROP_STATUS: {
                "select": {
                    "options": [
                        {"name": "todo", "color": "gray"},
                        {"name": "done", "color": "blue"},
                    ]
                }
            },
        },
    )
    db_id = response["id"]
    await set_config(DB_CONFIG_KEY, db_id)
    logger.info("Notion database created: %s", db_id)
    return db_id


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------
async def create_task(title: str, due_iso: Optional[str], priority: str = "medium") -> dict:
    db_id = await get_or_create_database()
    props: dict = {
        PROP_TITLE: {"title": [{"text": {"content": title}}]},
        PROP_PRIORITY: {"select": {"name": priority}},
        PROP_STATUS: {"select": {"name": "todo"}},
    }
    if due_iso:
        props[PROP_DUE] = {"date": {"start": due_iso}}

    page = await _notion.pages.create(
        parent={"database_id": db_id},
        properties=props,
    )
    logger.info("Task created: %s (%s)", title, page["id"])
    return page


async def mark_done(page_id: str) -> None:
    await _notion.pages.update(
        page_id=page_id,
        properties={PROP_STATUS: {"select": {"name": "done"}}},
    )


async def delete_task(page_id: str) -> None:
    await _notion.pages.update(page_id=page_id, archived=True)


# ---------------------------------------------------------------------------
# Smart list queries  (long-term memory retrieval)
# ---------------------------------------------------------------------------
async def list_tasks(filter_date: Optional[str] = None, include_done: bool = False) -> list[dict]:
    """List tasks, optionally filtered by exact date (YYYY-MM-DD)."""
    db_id = await get_or_create_database()
    conditions: list[dict] = []

    if not include_done:
        conditions.append({"property": PROP_STATUS, "select": {"equals": "todo"}})

    if filter_date:
        conditions.append({"property": PROP_DUE, "date": {"equals": filter_date}})

    query_filter = {"and": conditions} if len(conditions) > 1 else (conditions[0] if conditions else {})

    kwargs: dict = {
        "database_id": db_id,
        "sorts": [{"property": PROP_DUE, "direction": "ascending"}],
    }
    if query_filter:
        kwargs["filter"] = query_filter

    response = await _notion.databases.query(**kwargs)
    return response.get("results", [])


async def search_tasks_by_keyword(keyword: str, include_done: bool = False) -> list[dict]:
    """
    Search all tasks whose title contains the keyword.
    E.g. 'milk' → finds 'buy milk', 'get milk from store', etc.
    """
    db_id = await get_or_create_database()
    conditions: list[dict] = [
        {"property": PROP_TITLE, "title": {"contains": keyword}},
    ]
    if not include_done:
        conditions.append({"property": PROP_STATUS, "select": {"equals": "todo"}})

    response = await _notion.databases.query(
        database_id=db_id,
        filter={"and": conditions},
        sorts=[{"property": PROP_DUE, "direction": "ascending"}],
    )
    return response.get("results", [])


async def list_tasks_by_date_range(
    start_date: str,
    end_date: str,
    include_done: bool = True,
) -> list[dict]:
    """
    Return tasks due between start_date and end_date (YYYY-MM-DD).
    include_done=True so historical queries ("last 15 days") show completed tasks too.
    """
    db_id = await get_or_create_database()
    conditions: list[dict] = [
        {"property": PROP_DUE, "date": {"on_or_after":  start_date}},
        {"property": PROP_DUE, "date": {"on_or_before": end_date}},
    ]
    if not include_done:
        conditions.append({"property": PROP_STATUS, "select": {"equals": "todo"}})

    response = await _notion.databases.query(
        database_id=db_id,
        filter={"and": conditions},
        sorts=[{"property": PROP_DUE, "direction": "ascending"}],
    )
    return response.get("results", [])


async def find_task_by_title(title: str) -> Optional[dict]:
    """Return the first todo task whose title contains the string."""
    db_id = await get_or_create_database()
    response = await _notion.databases.query(
        database_id=db_id,
        filter={
            "and": [
                {"property": PROP_TITLE, "title": {"contains": title}},
                {"property": PROP_STATUS, "select": {"equals": "todo"}},
            ]
        },
    )
    results = response.get("results", [])
    return results[0] if results else None


async def get_due_soon(within_minutes: int = 15) -> list[dict]:
    now = datetime.now(timezone.utc)
    end = now + timedelta(minutes=within_minutes)
    db_id = await get_or_create_database()
    response = await _notion.databases.query(
        database_id=db_id,
        filter={
            "and": [
                {"property": PROP_STATUS, "select": {"equals": "todo"}},
                {"property": PROP_DUE, "date": {"on_or_after":  now.isoformat()}},
                {"property": PROP_DUE, "date": {"on_or_before": end.isoformat()}},
            ]
        },
    )
    return response.get("results", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def task_to_text(page: dict) -> str:
    props  = page.get("properties", {})
    title    = _get_title(props)
    due      = _get_date(props)
    priority = _get_select(props, PROP_PRIORITY)
    status   = _get_select(props, PROP_STATUS)
    status_icon = "✅" if status == "done" else "•"
    return f"{status_icon} {title}" + (f" — {due}" if due else "") + (f" [{priority}]" if priority else "")


def _get_title(props: dict) -> str:
    items = props.get(PROP_TITLE, {}).get("title", [])
    return "".join(i.get("plain_text", "") for i in items) or "Untitled"


def _get_date(props: dict) -> str:
    d = props.get(PROP_DUE, {}).get("date")
    return d["start"] if d else ""


def _get_select(props: dict, key: str) -> str:
    sel = props.get(key, {}).get("select")
    return sel["name"] if sel else ""
