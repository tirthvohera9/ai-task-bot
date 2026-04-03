"""
Task Engine: orchestrates routing → validation → Notion write → SQLite log.
Returns a human-readable response string for each action.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from db.database import log_behavior
from engine.router import route
from services.notion_service import (
    create_task,
    delete_task,
    find_task_by_title,
    list_tasks,
    mark_done,
    task_to_text,
)

logger = logging.getLogger(__name__)

# Minimum confidence to act without asking for clarification
CONFIDENCE_THRESHOLD = 0.6


async def handle_message(text: str, user_id: str) -> str:
    """Main entry point: text → response string."""
    intent = await route(text)

    if intent is None:
        return (
            "Sorry, I couldn't understand that. Try something like:\n"
            "• _Add meeting at 3pm tomorrow_\n"
            "• _Show today's tasks_\n"
            "• _Done with report_"
        )

    confidence = intent.get("confidence", 1.0)
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            f"I'm not sure what you mean (confidence: {confidence:.0%}). "
            "Could you rephrase? E.g. _Add [task] at [time]_"
        )

    action = intent["action"]

    if action == "add":
        return await _add(intent, user_id)
    elif action == "list":
        return await _list(intent)
    elif action == "delete":
        return await _delete(intent, user_id)
    elif action == "done":
        return await _done(intent, user_id)
    elif action == "update":
        return "Updates aren't supported yet. Delete and recreate the task."
    else:
        return f"Unknown action: {action}"


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

    # Validate datetime is in the future
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
    # Detect if user asked for today specifically
    source_text = intent.get("raw_text", "").lower()
    today_filter: Optional[str] = None

    if "today" in source_text or not any(
        w in source_text for w in ("tomorrow", "week", "all")
    ):
        today_filter = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    tasks = await list_tasks(filter_date=today_filter)
    if not tasks:
        label = "today" if today_filter else "upcoming"
        return f"No {label} tasks found. 🎉"

    lines = [f"📋 *{'Today' if today_filter else 'Upcoming'}'s tasks:*"]
    for page in tasks:
        lines.append(task_to_text(page))
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
