"""
Task Engine: routes intent → Notion query → human-readable response.
Notion is the long-term memory — all tasks are queried from there.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from db.database import log_behavior
from engine.router import route
from services.ai_service import general_chat
from services.notion_service import (
    create_task,
    delete_task,
    find_task_by_title,
    list_tasks,
    list_tasks_by_date_range,
    mark_done,
    search_tasks_by_keyword,
    task_to_text,
)

logger = logging.getLogger(__name__)
CONFIDENCE_THRESHOLD = 0.6


async def handle_message(text: str, user_id: str, history: Optional[list] = None) -> str:
    try:
        intent = await route(text, history=history)

        if intent is None or intent.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            return await general_chat(text, history=history)

        action = intent["action"]

        if action == "add":
            return await _add(intent, user_id)
        elif action == "list":
            return await _list(intent)
        elif action == "search":
            return await _search(intent)
        elif action == "delete":
            return await _delete(intent, user_id)
        elif action == "done":
            return await _done(intent, user_id)
        elif action == "update":
            return "Updates aren't supported yet — delete and recreate the task."
        else:
            return await general_chat(text, history=history)

    except Exception as exc:
        logger.exception("handle_message error for '%s': %s", text, exc)
        return await general_chat(text, history=history)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------
async def _add(intent: dict, user_id: str) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "What should I call this task? Please include a title."

    due = intent.get("datetime")
    if not due:
        return (
            f"When should I schedule *{title}*? "
            "Please add a time, e.g. _at 5pm_ or _tomorrow_."
        )

    try:
        dt = datetime.fromisoformat(due)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt <= datetime.now(timezone.utc):
            return f"That time is in the past. When would you like to schedule *{title}*?"
    except ValueError:
        return "I couldn't parse that date/time. Try _at 5pm_ or _tomorrow at 9am_."

    priority = intent.get("priority", "medium")
    await create_task(title, due, priority)
    await log_behavior(user_id, "created", title)

    due_fmt = dt.strftime("%b %d at %I:%M %p UTC")
    return f"✅ Task added: *{title}*\n📅 {due_fmt}\n🔔 Priority: {priority}"


async def _list(intent: dict) -> str:
    """Handle time-window list queries: today, tomorrow, last N days, all."""
    date_range = (intent.get("date_range") or "today").lower()
    include_done = intent.get("include_done", False)
    now = datetime.now(timezone.utc)

    if date_range == "today":
        tasks = await list_tasks(filter_date=now.strftime("%Y-%m-%d"))
        label = "Today"
    elif date_range == "tomorrow":
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        tasks = await list_tasks(filter_date=tomorrow)
        label = "Tomorrow"
    elif date_range in ("last_7_days", "last_15_days", "last_30_days"):
        days = int(date_range.split("_")[1])
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        tasks = await list_tasks_by_date_range(start, end, include_done=True)
        label = f"Last {days} days"
    elif date_range == "this_week":
        start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        tasks = await list_tasks_by_date_range(start, end, include_done=include_done)
        label = "This week"
    else:
        tasks = await list_tasks(include_done=include_done)
        label = "All"

    if not tasks:
        return f"No tasks found for *{label}*. 🎉"

    lines = [f"📋 *{label}'s tasks:*"]
    lines.extend(task_to_text(p) for p in tasks)
    return "\n".join(lines)


async def _search(intent: dict) -> str:
    """Handle keyword-based queries: 'when do I buy milk?', 'what about dentist?'"""
    keyword = (intent.get("keyword") or intent.get("title") or "").strip()
    include_done = intent.get("include_done", False)

    # Also apply date range if given
    date_range = (intent.get("date_range") or "").lower()
    now = datetime.now(timezone.utc)

    if not keyword:
        return await _list(intent)

    tasks = await search_tasks_by_keyword(keyword, include_done=include_done)

    # Filter by date range if specified
    if date_range and tasks:
        if date_range == "last_7_days":
            cutoff = now - timedelta(days=7)
        elif date_range == "last_15_days":
            cutoff = now - timedelta(days=15)
        elif date_range == "last_30_days":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = None

        if cutoff:
            filtered = []
            for t in tasks:
                due_str = t.get("properties", {}).get("Due", {}).get("date", {})
                if due_str and due_str.get("start"):
                    try:
                        due_dt = datetime.fromisoformat(due_str["start"])
                        if due_dt.tzinfo is None:
                            due_dt = due_dt.replace(tzinfo=timezone.utc)
                        if due_dt >= cutoff:
                            filtered.append(t)
                    except ValueError:
                        filtered.append(t)
                else:
                    filtered.append(t)
            tasks = filtered

    if not tasks:
        return f"No tasks found matching *{keyword}*."

    lines = [f"🔍 *Tasks matching \"{keyword}\":*"]
    lines.extend(task_to_text(p) for p in tasks)
    return "\n".join(lines)


async def _delete(intent: dict, user_id: str) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "Which task should I delete? Please include the task name."

    page = await find_task_by_title(title)
    if not page:
        return f"No task found matching *{title}*."

    await delete_task(page["id"])
    await log_behavior(user_id, "deleted", title)
    return f"🗑️ Deleted: *{title}*"


async def _done(intent: dict, user_id: str) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "Which task did you complete? Please include the task name."

    page = await find_task_by_title(title)
    if not page:
        return f"No active task found matching *{title}*."

    await mark_done(page["id"])
    await log_behavior(user_id, "completed", title)
    return f"✅ Marked done: *{title}* — great work!"
