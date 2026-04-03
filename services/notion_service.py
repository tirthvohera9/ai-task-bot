"""
Notion integration: database bootstrap, schema migration, task CRUD, smart queries.
Notion is the long-term memory — all tasks live here permanently.

Schema (v3):
  Name        — title
  Due         — date
  Priority    — select (high/medium/low)
  Status      — select (todo/done)
  Category    — select (work/personal/health/shopping/finance/travel/family/fitness/other)
  Notes       — rich_text  (context/purpose behind the task)
  Recurrence  — rich_text  (RRULE-like: "daily", "weekly:MON", "monthly:5", or empty)
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from notion_client import AsyncClient

from config import settings
from db.database import get_config, set_config

logger = logging.getLogger(__name__)
_notion = AsyncClient(auth=settings.NOTION_API_KEY)

DB_CONFIG_KEY  = "notion_database_id"
SCHEMA_VER_KEY = "notion_schema_version"
SCHEMA_VERSION = "v3"

PROP_TITLE      = "Name"
PROP_DUE        = "Due"
PROP_PRIORITY   = "Priority"
PROP_STATUS     = "Status"
PROP_CATEGORY   = "Category"
PROP_NOTES      = "Notes"
PROP_RECURRENCE = "Recurrence"

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
        await _ensure_schema_v3(db_id)
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
        PROP_DUE:   {"date": {}},
        PROP_PRIORITY: {
            "select": {"options": [
                {"name": "high",   "color": "red"},
                {"name": "medium", "color": "yellow"},
                {"name": "low",    "color": "green"},
            ]}
        },
        PROP_STATUS: {
            "select": {"options": [
                {"name": "todo", "color": "gray"},
                {"name": "done", "color": "blue"},
            ]}
        },
        PROP_CATEGORY: {
            "select": {"options": [
                {"name": "work",     "color": "blue"},
                {"name": "personal", "color": "purple"},
                {"name": "health",   "color": "green"},
                {"name": "shopping", "color": "orange"},
                {"name": "finance",  "color": "red"},
                {"name": "travel",   "color": "yellow"},
                {"name": "family",   "color": "pink"},
                {"name": "fitness",  "color": "brown"},
                {"name": "other",    "color": "gray"},
            ]}
        },
        PROP_NOTES:      {"rich_text": {}},
        PROP_RECURRENCE: {"rich_text": {}},
    }


async def _ensure_schema_v3(db_id: str) -> None:
    """Add Recurrence property to existing DB (idempotent, runs once)."""
    version = await get_config(SCHEMA_VER_KEY)
    if version == SCHEMA_VERSION:
        return
    try:
        await _notion.databases.update(
            database_id=db_id,
            properties={
                PROP_CATEGORY: {
                    "select": {"options": [
                        {"name": "work",     "color": "blue"},
                        {"name": "personal", "color": "purple"},
                        {"name": "health",   "color": "green"},
                        {"name": "shopping", "color": "orange"},
                        {"name": "finance",  "color": "red"},
                        {"name": "travel",   "color": "yellow"},
                        {"name": "family",   "color": "pink"},
                        {"name": "fitness",  "color": "brown"},
                        {"name": "other",    "color": "gray"},
                    ]}
                },
                PROP_NOTES:      {"rich_text": {}},
                PROP_RECURRENCE: {"rich_text": {}},
            },
        )
        await set_config(SCHEMA_VER_KEY, SCHEMA_VERSION)
        logger.info("Notion schema migrated to %s", SCHEMA_VERSION)
    except Exception as exc:
        logger.warning("Schema migration failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
async def create_task(
    title:      str,
    due_iso:    Optional[str],
    priority:   str = "medium",
    category:   Optional[str] = None,
    notes:      Optional[str] = None,
    recurrence: Optional[str] = None,
) -> dict:
    db_id = await get_or_create_database()
    props: dict = {
        PROP_TITLE:    {"title":  [{"text": {"content": title}}]},
        PROP_PRIORITY: {"select": {"name": priority}},
        PROP_STATUS:   {"select": {"name": "todo"}},
    }
    if due_iso:
        props[PROP_DUE] = {"date": {"start": due_iso}}
    if category and category in _CATEGORY_ICONS:
        props[PROP_CATEGORY] = {"select": {"name": category}}
    if notes:
        props[PROP_NOTES] = {"rich_text": [{"text": {"content": notes[:2000]}}]}
    if recurrence and recurrence != "none":
        props[PROP_RECURRENCE] = {"rich_text": [{"text": {"content": recurrence}}]}

    page = await _notion.pages.create(
        parent={"database_id": db_id},
        properties=props,
    )
    logger.info("Task created: '%s' cat=%s due=%s recurrence=%s", title, category, due_iso, recurrence)
    return page


async def update_task(page_id: str, **fields) -> None:
    """
    Update arbitrary task fields.
    Supported kwargs: title, due_iso, priority, category, notes, recurrence, status
    """
    props: dict = {}

    if "title" in fields and fields["title"]:
        props[PROP_TITLE] = {"title": [{"text": {"content": fields["title"]}}]}

    if "due_iso" in fields:
        val = fields["due_iso"]
        props[PROP_DUE] = {"date": {"start": val}} if val else {"date": None}

    if "priority" in fields and fields["priority"]:
        props[PROP_PRIORITY] = {"select": {"name": fields["priority"]}}

    if "category" in fields and fields["category"]:
        props[PROP_CATEGORY] = {"select": {"name": fields["category"]}}

    if "notes" in fields and fields["notes"]:
        props[PROP_NOTES] = {"rich_text": [{"text": {"content": str(fields["notes"])[:2000]}}]}

    if "recurrence" in fields:
        val = fields["recurrence"] or ""
        props[PROP_RECURRENCE] = {"rich_text": [{"text": {"content": val}}]}

    if "status" in fields and fields["status"]:
        props[PROP_STATUS] = {"select": {"name": fields["status"]}}

    if props:
        await _notion.pages.update(page_id=page_id, properties=props)
        logger.info("Task %s updated: %s", page_id, list(fields.keys()))


async def mark_done(page_id: str) -> None:
    await _notion.pages.update(
        page_id=page_id,
        properties={PROP_STATUS: {"select": {"name": "done"}}},
    )


async def delete_task(page_id: str) -> None:
    await _notion.pages.update(page_id=page_id, archived=True)


async def get_task_by_id(page_id: str) -> Optional[dict]:
    """Fetch a single Notion page by ID. Returns None on any error."""
    try:
        return await _notion.pages.retrieve(page_id=page_id)
    except Exception as exc:
        logger.warning("get_task_by_id(%s) failed: %s", page_id, exc)
        return None


# ---------------------------------------------------------------------------
# Smart queries
# ---------------------------------------------------------------------------
async def list_tasks(
    filter_date:  Optional[str] = None,
    include_done: bool = False,
) -> list[dict]:
    db_id = await get_or_create_database()
    conditions: list[dict] = []
    if not include_done:
        conditions.append({"property": PROP_STATUS, "select": {"equals": "todo"}})
    if filter_date:
        conditions.append({"property": PROP_DUE, "date": {"equals": filter_date}})

    kwargs: dict = {
        "database_id": db_id,
        "sorts": [{"property": PROP_DUE, "direction": "ascending"}],
    }
    f = _build_filter(conditions)
    if f:
        kwargs["filter"] = f
    return (await _notion.databases.query(**kwargs)).get("results", [])


async def list_tasks_by_date_range(
    start_date:   str,
    end_date:     str,
    include_done: bool = True,
) -> list[dict]:
    db_id = await get_or_create_database()
    conditions: list[dict] = [
        {"property": PROP_DUE, "date": {"on_or_after":  start_date}},
        {"property": PROP_DUE, "date": {"on_or_before": end_date}},
    ]
    if not include_done:
        conditions.append({"property": PROP_STATUS, "select": {"equals": "todo"}})
    return (await _notion.databases.query(
        database_id=db_id,
        filter={"and": conditions},
        sorts=[{"property": PROP_DUE, "direction": "ascending"}],
    )).get("results", [])


async def search_tasks_multi(
    keywords:     Optional[list[str]] = None,
    category:     Optional[str] = None,
    include_done: bool = False,
    date_range:   Optional[str] = None,
) -> list[dict]:
    """
    Search tasks matching ANY keyword OR the given category.
    Applies optional date range. Deduplicates and sorts by due date.
    """
    db_id = await get_or_create_database()
    all_results: dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    date_conditions = _date_range_conditions(date_range, now)
    status_cond     = [] if include_done else [{"property": PROP_STATUS, "select": {"equals": "todo"}}]

    for term in (keywords or [])[:6]:
        if not term or not term.strip():
            continue
        conds = [{"property": PROP_TITLE, "title": {"contains": term}}] + status_cond + date_conditions
        try:
            resp = await _notion.databases.query(
                database_id=db_id,
                filter=_build_filter(conds),
                sorts=[{"property": PROP_DUE, "direction": "ascending"}],
            )
            for p in resp.get("results", []):
                all_results[p["id"]] = p
        except Exception as exc:
            logger.warning("search term '%s' failed: %s", term, exc)

    if category and category in _CATEGORY_ICONS:
        conds = [{"property": PROP_CATEGORY, "select": {"equals": category}}] + status_cond + date_conditions
        try:
            resp = await _notion.databases.query(
                database_id=db_id,
                filter=_build_filter(conds),
                sorts=[{"property": PROP_DUE, "direction": "ascending"}],
            )
            for p in resp.get("results", []):
                all_results[p["id"]] = p
        except Exception as exc:
            logger.warning("category search '%s' failed: %s", category, exc)

    results = sorted(
        all_results.values(),
        key=lambda p: (
            (p.get("properties", {}).get(PROP_DUE, {}).get("date") or {}).get("start") or "9999"
        ),
    )
    return results


async def find_task_by_title(title: str) -> Optional[dict]:
    """Return the first todo task whose title contains the string."""
    db_id = await get_or_create_database()
    resp = await _notion.databases.query(
        database_id=db_id,
        filter={"and": [
            {"property": PROP_TITLE,  "title":  {"contains": title}},
            {"property": PROP_STATUS, "select": {"equals": "todo"}},
        ]},
    )
    results = resp.get("results", [])
    return results[0] if results else None


async def get_due_soon(within_minutes: int = 15) -> list[dict]:
    """Todo tasks due within the next N minutes."""
    now = datetime.now(timezone.utc)
    end = now + timedelta(minutes=within_minutes)
    db_id = await get_or_create_database()
    resp = await _notion.databases.query(
        database_id=db_id,
        filter={"and": [
            {"property": PROP_STATUS, "select": {"equals": "todo"}},
            {"property": PROP_DUE,    "date":   {"on_or_after":  now.isoformat()}},
            {"property": PROP_DUE,    "date":   {"on_or_before": end.isoformat()}},
        ]},
    )
    return resp.get("results", [])


async def list_recurring_tasks() -> list[dict]:
    """All todo tasks that have a recurrence rule set."""
    db_id = await get_or_create_database()
    resp = await _notion.databases.query(
        database_id=db_id,
        filter={"and": [
            {"property": PROP_STATUS, "select": {"equals": "done"}},
            {"property": PROP_RECURRENCE, "rich_text": {"is_not_empty": True}},
        ]},
    )
    return resp.get("results", [])


# ---------------------------------------------------------------------------
# Recurrence helpers
# ---------------------------------------------------------------------------
def next_recurrence_date(recurrence: str, from_dt: datetime) -> Optional[datetime]:
    """
    Given an RRULE-like string, return the next occurrence after from_dt.
    Formats: "daily", "weekly:MON", "monthly:5"
    """
    r = recurrence.lower().strip()
    if r == "daily":
        return from_dt + timedelta(days=1)

    if r.startswith("weekly:"):
        day_names = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        day_str = r.split(":")[1][:3]
        target = day_names.get(day_str)
        if target is None:
            return None
        days_ahead = (target - from_dt.weekday()) % 7 or 7
        return from_dt + timedelta(days=days_ahead)

    if r.startswith("monthly:"):
        try:
            target_day = int(r.split(":")[1])
            next_month = from_dt.month % 12 + 1
            yr = from_dt.year + (1 if from_dt.month == 12 else 0)
            return from_dt.replace(year=yr, month=next_month, day=target_day)
        except (ValueError, TypeError):
            return None

    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def task_to_text(page: dict, user_tz: str = "UTC") -> str:
    """Format a Notion page as a readable task string."""
    from utils.datetime_parser import format_local

    props      = page.get("properties", {})
    title      = _get_title(props)
    due        = _get_date(props)
    priority   = _get_select(props, PROP_PRIORITY)
    status     = _get_select(props, PROP_STATUS)
    category   = _get_select(props, PROP_CATEGORY)
    notes      = _get_rich_text(props, PROP_NOTES)
    recurrence = _get_rich_text(props, PROP_RECURRENCE)

    done     = status == "done"
    cat_icon = _CATEGORY_ICONS.get(category, "") if category else ""

    # Title — strike-through style for done tasks
    line = f"~{title}~" if done else f"*{title}*"
    if cat_icon:
        line += f" {cat_icon}"

    if due:
        line += f" — {format_local(due, user_tz)}"

    if priority == "high":
        line += " 🔴"
    elif priority == "low":
        line += " 🟢"

    if recurrence and recurrence != "none":
        line += " ♻️"

    if notes:
        line += f"\n   ↳ _{notes}_"

    return line


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_filter(conditions: list[dict]) -> dict:
    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"and": conditions}


def _date_range_conditions(date_range: Optional[str], now: datetime) -> list[dict]:
    if not date_range:
        return []
    if date_range == "today":
        d = now.strftime("%Y-%m-%d")
        return [{"property": PROP_DUE, "date": {"equals": d}}]
    if date_range == "tomorrow":
        d = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        return [{"property": PROP_DUE, "date": {"equals": d}}]
    if date_range == "this_week":
        start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        end   = (now + timedelta(days=6 - now.weekday())).strftime("%Y-%m-%d")
        return [
            {"property": PROP_DUE, "date": {"on_or_after":  start}},
            {"property": PROP_DUE, "date": {"on_or_before": end}},
        ]
    if date_range in ("last_7_days", "last_15_days", "last_30_days"):
        days  = int(date_range.split("_")[1])
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = now.strftime("%Y-%m-%d")
        return [
            {"property": PROP_DUE, "date": {"on_or_after":  start}},
            {"property": PROP_DUE, "date": {"on_or_before": end}},
        ]
    return []


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
