"""
Task Engine: routes intent → Notion query → human-readable response.
Notion is the long-term memory — all tasks are queried from there.

Smart behaviours:
  - Multi-keyword search: "what do I need to buy?" searches ["buy","groceries","shopping"...]
  - Category-aware: stores and queries category (health/work/shopping/finance/...)
  - Purpose-aware: stores notes (why behind a task), retrieves them in answers
  - Synthesized responses: AI generates natural answers instead of raw task dumps
  - Conversation history: all operations are context-aware
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from db.database import log_behavior
from engine.router import route
from services.ai_service import general_chat, synthesize_response
from services.notion_service import (
    create_task,
    delete_task,
    find_task_by_title,
    list_tasks,
    list_tasks_by_date_range,
    mark_done,
    search_tasks_multi,
    task_to_text,
)

logger = logging.getLogger(__name__)
CONFIDENCE_THRESHOLD = 0.55


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
            return await _search(intent, original_text=text, history=history)
        elif action == "delete":
            return await _delete(intent, user_id)
        elif action == "done":
            return await _done(intent, user_id)
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
            "Please add a time — e.g. _at 5pm_, _tomorrow_, or _next Monday_."
        )

    try:
        dt = datetime.fromisoformat(due)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt <= datetime.now(timezone.utc):
            return f"That time is already in the past. When would you like to schedule *{title}*?"
    except ValueError:
        return "I couldn't parse that date/time. Try _tomorrow at 5pm_ or _next Monday at 9am_."

    priority = intent.get("priority", "medium")
    category = intent.get("category")
    notes    = intent.get("notes")

    await create_task(title, due, priority=priority, category=category, notes=notes)
    await log_behavior(user_id, "created", title)

    due_fmt = dt.strftime("%-d %b at %-I:%M %p UTC")
    cat_line = f"\n🏷️ Category: {category}" if category else ""
    note_line = f"\n📝 _{notes}_" if notes else ""
    return f"✅ Added: *{title}*\n📅 {due_fmt}{cat_line}{note_line}"


async def _list(intent: dict) -> str:
    """Handle time-window list queries: today, tomorrow, this week, last N days, all."""
    date_range   = (intent.get("date_range") or "today").lower()
    include_done = intent.get("include_done", False)
    now = datetime.now(timezone.utc)

    if date_range == "today":
        tasks = await list_tasks(filter_date=now.strftime("%Y-%m-%d"), include_done=include_done)
        label = "Today"

    elif date_range == "tomorrow":
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        tasks = await list_tasks(filter_date=tomorrow, include_done=include_done)
        label = "Tomorrow"

    elif date_range in ("last_7_days", "last_15_days", "last_30_days"):
        days  = int(date_range.split("_")[1])
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = now.strftime("%Y-%m-%d")
        tasks = await list_tasks_by_date_range(start, end, include_done=True)
        label = f"Last {days} days"

    elif date_range == "this_week":
        # Mon → Sun of the current calendar week
        week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        week_end   = (now + timedelta(days=6 - now.weekday())).strftime("%Y-%m-%d")
        tasks = await list_tasks_by_date_range(week_start, week_end, include_done=include_done)
        label = "This week"

    elif date_range == "all":
        tasks = await list_tasks(include_done=True)
        label = "All tasks"

    else:
        tasks = await list_tasks(filter_date=now.strftime("%Y-%m-%d"))
        label = "Today"

    if not tasks:
        empties = {
            "Today":    "You have nothing scheduled for today. 🎉",
            "Tomorrow": "Nothing scheduled for tomorrow yet.",
        }
        return empties.get(label, f"No tasks found for *{label}*.")

    lines = [f"📋 *{label}:*"]
    lines.extend(task_to_text(p) for p in tasks)
    return "\n".join(lines)


async def _search(
    intent: dict,
    original_text: str = "",
    history: Optional[list] = None,
) -> str:
    """
    Handle any question-style query using multi-keyword + category search in Notion,
    then synthesise a natural language answer with AI.

    Examples:
      "when do I have to buy milk?"
      "any doctor appointments?"
      "what do I need to buy?"
      "anything work-related this week?"
      "did I set reminders in the last 15 days?"
    """
    # Gather all search terms
    keywords: list[str] = list(intent.get("keywords") or [])
    keyword = (intent.get("keyword") or intent.get("title") or "").strip()
    if keyword and keyword not in keywords:
        keywords.insert(0, keyword)

    category   = intent.get("category")
    date_range = intent.get("date_range") or None
    include_done = intent.get("include_done", False)

    # If we have nothing to search on, fall back to listing
    if not keywords and not category:
        return await _list(intent)

    tasks = await search_tasks_multi(
        keywords=keywords,
        category=category,
        include_done=include_done,
        date_range=date_range,
    )

    task_texts = [task_to_text(p) for p in tasks]

    # Use AI to synthesize a natural response to the original question
    question = original_text or keyword or "query"
    return await synthesize_response(question, task_texts, history=history)


async def _delete(intent: dict, user_id: str) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "Which task should I delete? Please include the task name."

    page = await find_task_by_title(title)
    if not page:
        return f"No task found matching *{title}*. Maybe it was already removed?"

    await delete_task(page["id"])
    await log_behavior(user_id, "deleted", title)
    return f"🗑️ Deleted: *{title}*"


async def _done(intent: dict, user_id: str) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "Which task did you complete? Please include the task name."

    page = await find_task_by_title(title)
    if not page:
        # Try searching with keywords in case title is partially matched
        keywords = [w for w in title.split() if len(w) > 3]
        if keywords:
            results = await search_tasks_multi(keywords=keywords, include_done=False)
            if results:
                page = results[0]

    if not page:
        return f"No active task found matching *{title}*."

    actual_title = _extract_title(page)
    await mark_done(page["id"])
    await log_behavior(user_id, "completed", actual_title)
    return f"✅ Marked done: *{actual_title}* — great work! 🎉"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_title(page: dict) -> str:
    items = page.get("properties", {}).get("Name", {}).get("title", [])
    return "".join(i.get("plain_text", "") for i in items) or "task"
