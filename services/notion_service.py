"""
Notion integration: database bootstrap + task CRUD + smart queries.
Notion is the long-term memory — all tasks live here permanently.

Schema (v2):
  Name       — title
  Due        — date
  Priority   — select (high/medium/low)
  Status     — select (todo/done)
  Category   — select (work/personal/health/shopping/finance/travel/family/fitness/other)
  Notes      — rich_text (context/purpose behind the task)
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from notion_client import AsyncClient

from config import settings
from db.database import get_config, set_config

logger = logging.getLogger(__name__)
_notion = AsyncClient(auth=settings.NOTION_API_KEY)

DB_CONFIG_KEY    = "notion_database_id"
SCHEMA_VER_KEY   = "notion_schema_version"
SCHEMA_VERSION   = "v2"

PROP_TITLE    = "Name"
PROP_DUE      = "Due"
PROP_PRIORITY = "Priority"
PROP_STATUS   = "Status"
PROP_CATEGORY = "Category"
PROP_NOTES    = "Notes"

_CATEGORY_ICONS = {
    "work":     "💼",
    "personal": "👤",
    "health":   "🏥",
    "shopping": "🛒",
    "finance":  "💰",
    "travel":   "✈️",
    "family":   "👨‍👩‍👧",
    "fitness":  "💪",
    "other":    "📌",
}


# ---------------------------------------------------------------------------
# Database bootstrap + schema migration
# ---------------------------------------------------------------------------
async def get_or_create_database() -> str:
    db_id = await get_config(DB_CONFIG_KEY)
    if db_id:
        # Ensure schema is up to date (runs once, cached in Redis)
        await _ensure_schema_v2(db_id)
        return db_id

    logger.info("Creating Notion tasks database…")
    response = await _notion.databases.create(
        parent={"type": "page_id", "page_id": settings.NOTION_PARENT_PAGE_ID},
        title=[{"type": "text", "text": {"content": "Tasks"}}],
        properties=_full_schema(),
    )
    db_id = response["id"]
    await set_config(DB_CONFIG_KEY, db_id)
    await set_config(SCHEMA_VER_KEY, SCHEMA_VERSION)
    logger.info("Notion database created: %s", db_id)
    return db_id


def _full_schema() -> dict:
    return {
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
        PROP_CATEGORY: {
            "select": {
                "options": [
                    {"name": "work",     "color": "blue"},
                    {"name": "personal", "color": "purple"},
                    {"name": "health",   "color": "green"},
                    {"name": "shopping", "color": "orange"},
                    {"name": "finance",  "color": "red"},
                    {"name": "travel",   "color": "yellow"},
                    {"name": "family",   "color": "pink"},
                    {"name": "fitness",  "color": "brown"},
                    {"name": "other",    "color": "gray"},
                ]
            }
        },
        PROP_NOTES: {"rich_text": {}},
    }


async def _ensure_schema_v2(db_id: str) -> None:
    """Add Category + Notes properties to an existing DB (idempotent)."""
    version = await get_config(SCHEMA_VER_KEY)
    if version == SCHEMA_VERSION:
        return
    try:
        await _notion.databases.update(
            database_id=db_id,
            properties={
                PROP_CATEGORY: {
                    "select": {
                        "options": [
                            {"name": "work",     "color": "blue"},
                            {"name": "personal", "color": "purple"},
                            {"name": "health",   "color": "green"},
                            {"name": "shopping", "color": "orange"},
                            {"name": "finance",  "color": "red"},
                            {"name": "travel",   "color": "yellow"},
                            {"name": "family",   "color": "pink"},
                            {"name": "fitness",  "color": "brown"},
                            {"name": "other",    "color": "gray"},
                        ]
                    }
                },
                PROP_NOTES: {"rich_text": {}},
            },
        )
        await set_config(SCHEMA_VER_KEY, SCHEMA_VERSION)
        logger.info("Notion schema migrated to %s", SCHEMA_VERSION)
    except Exception as exc:
        logger.warning("Schema migration failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------
async def create_task(
    title: str,
    due_iso: Optional[str],
    priority: str = "medium",
    category: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    db_id = await get_or_create_database()
    props: dict = {
        PROP_TITLE:    {"title": [{"text": {"content": title}}]},
        PROP_PRIORITY: {"select": {"name": priority}},
        PROP_STATUS:   {"select": {"name": "todo"}},
    }
    if due_iso:
        props[PROP_DUE] = {"date": {"start": due_iso}}
    if category and category in _CATEGORY_ICONS:
        props[PROP_CATEGORY] = {"select": {"name": category}}
    if notes:
        props[PROP_NOTES] = {"rich_text": [{"text": {"content": notes[:2000]}}]}

    page = await _notion.pages.create(
        parent={"database_id": db_id},
        properties=props,
    )
    logger.info("Task created: '%s' cat=%s due=%s", title, category, due_iso)
    return page


async def mark_done(page_id: str) -> None:
    await _notion.pages.update(
        page_id=page_id,
        properties={PROP_STATUS: {"select": {"name": "done"}}},
    )


async def delete_task(page_id: str) -> None:
    await _notion.pages.update(page_id=page_id, archived=True)


# ---------------------------------------------------------------------------
# Smart queries — long-term memory retrieval
# ---------------------------------------------------------------------------
async def list_tasks(
    filter_date: Optional[str] = None,
    include_done: bool = False,
) -> list[dict]:
    """List tasks, optionally filtered by exact date (YYYY-MM-DD)."""
    db_id = await get_or_create_database()
    conditions: list[dict] = []

    if not include_done:
        conditions.append({"property": PROP_STATUS, "select": {"equals": "todo"}})
    if filter_date:
        conditions.append({"property": PROP_DUE, "date": {"equals": filter_date}})

    query_filter = _build_filter(conditions)
    kwargs: dict = {
        "database_id": db_id,
        "sorts": [{"property": PROP_DUE, "direction": "ascending"}],
    }
    if query_filter:
        kwargs["filter"] = query_filter

    response = await _notion.databases.query(**kwargs)
    return response.get("results", [])


async def list_tasks_by_date_range(
    start_date: str,
    end_date: str,
    include_done: bool = True,
) -> list[dict]:
    """Return tasks due between start_date and end_date (YYYY-MM-DD)."""
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


async def search_tasks_by_keyword(
    keyword: str,
    include_done: bool = False,
) -> list[dict]:
    """Search tasks whose title contains the keyword."""
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


async def search_tasks_multi(
    keywords: Optional[list[str]] = None,
    category: Optional[str] = None,
    include_done: bool = False,
    date_range: Optional[str] = None,
) -> list[dict]:
    """
    Search tasks matching ANY of the keywords OR the given category.
    Also applies optional date range filtering.
    Deduplicates results by page ID.
    """
    db_id = await get_or_create_database()
    all_results: dict[str, dict] = {}

    # Build date range conditions
    now = datetime.now(timezone.utc)
    date_conditions: list[dict] = []
    if date_range:
        if date_range == "today":
            date_conditions = [{"property": PROP_DUE, "date": {"equals": now.strftime("%Y-%m-%d")}}]
        elif date_range == "tomorrow":
            tom = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            date_conditions = [{"property": PROP_DUE, "date": {"equals": tom}}]
        elif date_range == "this_week":
            start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
            end = (now + timedelta(days=6 - now.weekday())).strftime("%Y-%m-%d")
            date_conditions = [
                {"property": PROP_DUE, "date": {"on_or_after": start}},
                {"property": PROP_DUE, "date": {"on_or_before": end}},
            ]
        elif date_range in ("last_7_days", "last_15_days", "last_30_days"):
            days = int(date_range.split("_")[1])
            start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
            date_conditions = [
                {"property": PROP_DUE, "date": {"on_or_after": start}},
                {"property": PROP_DUE, "date": {"on_or_before": end}},
            ]

    status_condition = (
        [] if include_done
        else [{"property": PROP_STATUS, "select": {"equals": "todo"}}]
    )

    # Search by each keyword separately (Notion OR is available but limited)
    search_terms = list(keywords or [])

    for term in search_terms[:6]:  # cap at 6 API calls
        if not term or not term.strip():
            continue
        conditions = (
            [{"property": PROP_TITLE, "title": {"contains": term}}]
            + status_condition
            + date_conditions
        )
        try:
            resp = await _notion.databases.query(
                database_id=db_id,
                filter=_build_filter(conditions),
                sorts=[{"property": PROP_DUE, "direction": "ascending"}],
            )
            for page in resp.get("results", []):
                all_results[page["id"]] = page
        except Exception as exc:
            logger.warning("search term '%s' failed: %s", term, exc)

    # Also search by category if provided
    if category and category in _CATEGORY_ICONS:
        conditions = (
            [{"property": PROP_CATEGORY, "select": {"equals": category}}]
            + status_condition
            + date_conditions
        )
        try:
            resp = await _notion.databases.query(
                database_id=db_id,
                filter=_build_filter(conditions),
                sorts=[{"property": PROP_DUE, "direction": "ascending"}],
            )
            for page in resp.get("results", []):
                all_results[page["id"]] = page
        except Exception as exc:
            logger.warning("category search '%s' failed: %s", category, exc)

    # Sort combined results by due date
    results = list(all_results.values())
    results.sort(key=lambda p: (
        p.get("properties", {}).get(PROP_DUE, {}).get("date", {}) or {}
    ).get("start") or "9999")

    return results


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
    """Return todo tasks due within the next N minutes."""
    now = datetime.now(timezone.utc)
    end = now + timedelta(minutes=within_minutes)
    db_id = await get_or_create_database()
    response = await _notion.databases.query(
        database_id=db_id,
        filter={
            "and": [
                {"property": PROP_STATUS, "select": {"equals": "todo"}},
                {"property": PROP_DUE,    "date": {"on_or_after":  now.isoformat()}},
                {"property": PROP_DUE,    "date": {"on_or_before": end.isoformat()}},
            ]
        },
    )
    return response.get("results", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_filter(conditions: list[dict]) -> dict:
    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"and": conditions}


def task_to_text(page: dict) -> str:
    """Format a Notion page as a readable task string."""
    props    = page.get("properties", {})
    title    = _get_title(props)
    due      = _get_date(props)
    priority = _get_select(props, PROP_PRIORITY)
    status   = _get_select(props, PROP_STATUS)
    category = _get_select(props, PROP_CATEGORY)
    notes    = _get_rich_text(props, PROP_NOTES)

    status_icon = "✅" if status == "done" else "•"
    cat_icon    = _CATEGORY_ICONS.get(category, "") if category else ""

    # Build the main line
    prefix = f"{status_icon} {cat_icon}".strip()
    line   = f"{prefix} *{title}*"

    # Format due date nicely
    if due:
        try:
            dt = datetime.fromisoformat(due)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            formatted = dt.strftime("%-d %b at %-I:%M %p UTC")
        except Exception:
            formatted = due
        line += f" — {formatted}"

    if priority and priority != "medium":
        priority_labels = {"high": "🔴 high", "low": "🟢 low"}
        line += f" [{priority_labels.get(priority, priority)}]"

    if notes:
        line += f"\n  ↳ _{notes}_"

    return line


def _get_title(props: dict) -> str:
    items = props.get(PROP_TITLE, {}).get("title", [])
    return "".join(i.get("plain_text", "") for i in items) or "Untitled"


def _get_date(props: dict) -> str:
    d = props.get(PROP_DUE, {}).get("date")
    return d["start"] if d else ""


def _get_select(props: dict, key: str) -> str:
    sel = props.get(key, {}).get("select")
    return sel["name"] if sel else ""


def _get_rich_text(props: dict, key: str) -> str:
    items = props.get(key, {}).get("rich_text", [])
    return "".join(i.get("plain_text", "") for i in items)
