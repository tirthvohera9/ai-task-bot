"""
Task Engine — deterministic execution layer.

Principle:
  - LLM parses intent (what the user wants)
  - This engine executes it deterministically (no LLM in the execution path)
  - State (pending confirmations, history) is stored in Redis, never in LLM

Flow:
  handle_message()
    ├─ Check pending confirmation → yes/no/cancel handling
    ├─ Route intent → action handler
    │     add    → validate → ask confirmation → save to Notion
    │     list   → Notion date query → formatted output
    │     search → Notion multi-keyword → AI synthesized answer
    │     update → find task → deterministic field update
    │     done   → fuzzy find → mark done → spawn next if recurring
    │     delete → fuzzy find → archive
    └─ Fallback to general_chat()
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from db.database import (
    clear_pending_task,
    get_behavior_patterns,
    get_pending_task,
    get_user_timezone,
    log_behavior,
    record_behavior_pattern,
    set_pending_task,
)
from engine.router import route
from services.ai_service import general_chat, synthesize_response
from services.notion_service import (
    create_task,
    delete_task,
    find_task_by_title,
    list_tasks,
    list_tasks_by_date_range,
    mark_done,
    next_recurrence_date,
    search_tasks_multi,
    task_to_text,
    update_task,
    _get_rich_text,
    PROP_RECURRENCE,
    PROP_DUE,
    PROP_PRIORITY,
    PROP_TITLE,
)
from utils.datetime_parser import extract_datetime, format_local

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.55

# Phrases the user can send to confirm or cancel a pending task
_YES_WORDS  = {"yes", "y", "yep", "yeah", "sure", "ok", "okay", "confirm", "go", "do it", "save", "add it"}
_NO_WORDS   = {"no", "n", "nope", "cancel", "stop", "abort", "nevermind", "never mind", "don't", "dont"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def handle_message(text: str, user_id: str, history: Optional[list] = None) -> str:
    user_tz = await get_user_timezone(user_id)

    # ── 1. Handle pending confirmation first ─────────────────────────────
    pending = await get_pending_task(user_id)
    if pending:
        lowered = text.strip().lower()
        if lowered in _YES_WORDS:
            return await _execute_confirmed_task(pending, user_id, user_tz)
        if lowered in _NO_WORDS:
            await clear_pending_task(user_id)
            return "❌ Cancelled. What else can I help you with?"
        # Not a yes/no — clear pending and process the new message normally
        await clear_pending_task(user_id)

    # ── 2. Route and dispatch ─────────────────────────────────────────────
    try:
        intent = await route(text, history=history)

        if intent is None or intent.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            return await general_chat(text, history=history)

        action = intent["action"]

        if action == "add":
            return await _add(intent, user_id, user_tz, history)
        elif action == "list":
            return await _list(intent, user_tz)
        elif action == "search":
            return await _search(intent, original_text=text, history=history, user_tz=user_tz)
        elif action == "update":
            return await _update(intent, user_id, user_tz)
        elif action == "delete":
            return await _delete(intent, user_id)
        elif action == "done":
            return await _done(intent, user_id, user_tz)
        else:
            return await general_chat(text, history=history)

    except Exception as exc:
        logger.exception("handle_message error for '%s': %s", text, exc)
        return await general_chat(text, history=history)


# ---------------------------------------------------------------------------
# ADD with confirmation
# ---------------------------------------------------------------------------
async def _add(intent: dict, user_id: str, user_tz: str, history: Optional[list]) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "What should I call this task? Please include a title."

    due       = intent.get("datetime")
    priority  = intent.get("priority", "medium")
    category  = intent.get("category")
    notes     = intent.get("notes")
    recurrence = intent.get("recurrence") or "none"

    # If no datetime, ask for one
    if not due:
        # Try to suggest a time based on past behavior
        suggestion = await _suggest_time(user_id, category)
        hint = f" (e.g. _tomorrow at {suggestion}_)" if suggestion else " (e.g. _tomorrow at 5pm_)"
        return (
            f"When should I schedule *{title}*?{hint}"
        )

    # Validate datetime
    try:
        dt = datetime.fromisoformat(due)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt <= datetime.now(timezone.utc):
            return f"That time is already in the past. When would you like to schedule *{title}*?"
    except ValueError:
        return "I couldn't parse that date/time. Try _tomorrow at 5pm_ or _next Monday at 9am_."

    # Build confirmation message
    local_time  = format_local(due, user_tz)
    cat_line    = f" [{category}]" if category else ""
    recur_line  = f"\n♻️ Repeats: _{recurrence}_" if recurrence and recurrence != "none" else ""
    note_line   = f"\n📝 _{notes}_" if notes else ""

    task_data = {
        "title": title, "due": due, "priority": priority,
        "category": category, "notes": notes, "recurrence": recurrence,
    }
    await set_pending_task(user_id, task_data)

    return (
        f"Confirm adding:\n"
        f"📌 *{title}*{cat_line}\n"
        f"📅 {local_time}\n"
        f"🔔 Priority: {priority}{recur_line}{note_line}\n\n"
        f"Reply *yes* to save or *no* to cancel."
    )


async def _execute_confirmed_task(pending: dict, user_id: str, user_tz: str) -> str:
    """Actually save a confirmed pending task to Notion."""
    await clear_pending_task(user_id)
    title      = pending["title"]
    due        = pending.get("due")
    priority   = pending.get("priority", "medium")
    category   = pending.get("category")
    notes      = pending.get("notes")
    recurrence = pending.get("recurrence", "none")

    await create_task(
        title, due,
        priority=priority,
        category=category,
        notes=notes,
        recurrence=recurrence if recurrence != "none" else None,
    )
    await log_behavior(user_id, "created", title)

    # Record behavioral pattern for future suggestions
    if due:
        try:
            dt = datetime.fromisoformat(due).astimezone(timezone.utc)
            await record_behavior_pattern(user_id, category, dt.hour)
        except Exception:
            pass

    local_time  = format_local(due, user_tz) if due else "No time set"
    cat_line    = f"\n🏷️ {category}" if category else ""
    recur_line  = f"\n♻️ Repeats: _{recurrence}_" if recurrence and recurrence != "none" else ""
    note_line   = f"\n📝 _{notes}_" if notes else ""

    return f"✅ Saved: *{title}*\n📅 {local_time}{cat_line}{recur_line}{note_line}"


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------
async def _list(intent: dict, user_tz: str) -> str:
    date_range   = (intent.get("date_range") or "today").lower()
    include_done = intent.get("include_done", False)
    now          = datetime.now(timezone.utc)

    if date_range == "today":
        tasks = await list_tasks(filter_date=now.strftime("%Y-%m-%d"), include_done=include_done)
        label = "Today"
    elif date_range == "tomorrow":
        from datetime import timedelta as _td
        tasks = await list_tasks(
            filter_date=(now + _td(days=1)).strftime("%Y-%m-%d"),
            include_done=include_done,
        )
        label = "Tomorrow"
    elif date_range in ("last_7_days", "last_15_days", "last_30_days"):
        from datetime import timedelta
        days  = int(date_range.split("_")[1])
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = now.strftime("%Y-%m-%d")
        tasks = await list_tasks_by_date_range(start, end, include_done=True)
        label = f"Last {days} days"
    elif date_range == "this_week":
        from datetime import timedelta
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
        return {"Today": "Nothing scheduled for today 🎉", "Tomorrow": "Nothing scheduled for tomorrow yet."}.get(
            label, f"No tasks found for *{label}*."
        )

    lines = [f"📋 *{label}:*"]
    lines.extend(task_to_text(p, user_tz) for p in tasks)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SEARCH — multi-keyword + category + AI synthesis
# ---------------------------------------------------------------------------
async def _search(
    intent: dict,
    original_text: str = "",
    history: Optional[list] = None,
    user_tz: str = "UTC",
) -> str:
    keywords: list[str] = list(intent.get("keywords") or [])
    keyword = (intent.get("keyword") or intent.get("title") or "").strip()
    if keyword and keyword not in keywords:
        keywords.insert(0, keyword)

    category     = intent.get("category")
    date_range   = intent.get("date_range") or None
    include_done = intent.get("include_done", False)

    if not keywords and not category:
        return await _list(intent, user_tz)

    tasks = await search_tasks_multi(
        keywords=keywords,
        category=category,
        include_done=include_done,
        date_range=date_range,
    )

    task_texts = [task_to_text(p, user_tz) for p in tasks]
    return await synthesize_response(original_text or keyword or "query", task_texts, history=history)


# ---------------------------------------------------------------------------
# UPDATE — deterministic field editing
# ---------------------------------------------------------------------------
async def _update(intent: dict, user_id: str, user_tz: str) -> str:
    title        = (intent.get("title") or "").strip()
    update_field = (intent.get("update_field") or "").lower()
    update_value = (intent.get("update_value") or "").strip()

    if not title:
        return "Which task should I update? Please include the task name."
    if not update_field or not update_value:
        return "What should I change? (e.g. 'move to 6pm', 'mark as high priority')"

    page = await find_task_by_title(title)
    if not page:
        # Try keyword search
        results = await search_tasks_multi(keywords=[w for w in title.split() if len(w) > 3])
        if results:
            page = results[0]

    if not page:
        return f"No active task found matching *{title}*."

    page_id     = page["id"]
    actual_title = _get_title_from_page(page)
    kwargs: dict = {}

    if update_field == "datetime":
        new_dt = extract_datetime(update_value, user_tz=user_tz)
        if not new_dt:
            return f"I couldn't parse '{update_value}' as a date/time. Try 'tomorrow at 6pm'."
        kwargs["due_iso"] = new_dt
        local_time = format_local(new_dt, user_tz)
        msg = f"📅 Rescheduled *{actual_title}* to {local_time}"

    elif update_field == "priority":
        p = update_value.lower()
        if p not in {"low", "medium", "high"}:
            return "Priority must be low, medium, or high."
        kwargs["priority"] = p
        msg = f"🔔 Updated priority of *{actual_title}* to {p}"

    elif update_field == "title":
        kwargs["title"] = update_value
        msg = f"✏️ Renamed to *{update_value}*"

    elif update_field == "status":
        s = update_value.lower()
        if s in {"done", "complete", "completed", "finished"}:
            kwargs["status"] = "done"
            msg = f"✅ Marked *{actual_title}* as done"
        else:
            kwargs["status"] = "todo"
            msg = f"🔄 Marked *{actual_title}* as todo"
    else:
        return f"I can update: datetime, priority, title, or status."

    await update_task(page_id, **kwargs)
    await log_behavior(user_id, "updated", actual_title)
    return msg


# ---------------------------------------------------------------------------
# DONE — mark complete + spawn next if recurring
# ---------------------------------------------------------------------------
async def _done(intent: dict, user_id: str, user_tz: str) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "Which task did you complete? Please include the task name."

    page = await find_task_by_title(title)
    if not page:
        results = await search_tasks_multi(keywords=[w for w in title.split() if len(w) > 3])
        if results:
            page = results[0]

    if not page:
        return f"No active task found matching *{title}*."

    page_id      = page["id"]
    actual_title = _get_title_from_page(page)
    props        = page.get("properties", {})
    recurrence   = _get_rich_text(props, PROP_RECURRENCE)

    await mark_done(page_id)
    await log_behavior(user_id, "completed", actual_title)

    follow_up = ""
    # If recurring, spawn the next instance
    if recurrence and recurrence.strip() and recurrence.strip() != "none":
        due_str = (props.get(PROP_DUE, {}).get("date") or {}).get("start", "")
        try:
            from_dt = datetime.fromisoformat(due_str) if due_str else datetime.now(timezone.utc)
            if from_dt.tzinfo is None:
                from_dt = from_dt.replace(tzinfo=timezone.utc)
            next_dt = next_recurrence_date(recurrence, from_dt)
            if next_dt:
                next_iso   = next_dt.isoformat()
                next_local = format_local(next_iso, user_tz)
                # Carry over category, notes, priority from done task
                category_val = _get_select_from_page(page, "Category")
                priority_val = _get_select_from_page(page, "Priority") or "medium"
                notes_val    = _get_rich_text(props, "Notes")
                await create_task(
                    actual_title, next_iso,
                    priority=priority_val,
                    category=category_val or None,
                    notes=notes_val or None,
                    recurrence=recurrence,
                )
                follow_up = f"\n♻️ Next occurrence created: {next_local}"
        except Exception as exc:
            logger.warning("Recurrence expansion failed: %s", exc)

    return f"✅ Done: *{actual_title}* — great work! 🎉{follow_up}"


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Behavioral intelligence — suggest times based on past patterns
# ---------------------------------------------------------------------------
async def _suggest_time(user_id: str, category: Optional[str]) -> Optional[str]:
    """
    Return a suggested time string based on the user's past task patterns
    for this category. E.g. "5pm" if the user usually adds shopping at 17:00.
    """
    if not category:
        return None
    try:
        patterns = await get_behavior_patterns(user_id)
        hour_key = f"hours_{category}"
        hours    = patterns.get(hour_key, {})
        if not hours:
            return None
        # Most frequent hour
        best_hour = max(hours, key=lambda h: hours[h])
        h         = int(best_hour)
        if h == 0:
            return "12am"
        if h < 12:
            return f"{h}am"
        if h == 12:
            return "12pm"
        return f"{h - 12}pm"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_title_from_page(page: dict) -> str:
    items = page.get("properties", {}).get(PROP_TITLE, {}).get("title", [])
    return "".join(i.get("plain_text", "") for i in items) or "task"


def _get_select_from_page(page: dict, prop_name: str) -> str:
    sel = page.get("properties", {}).get(prop_name, {}).get("select")
    return sel["name"] if sel else ""
