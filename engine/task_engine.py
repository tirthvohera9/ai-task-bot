"""
Task Engine — deterministic execution layer.

Principle:
  - LLM parses intent (what the user wants)
  - This engine executes it deterministically (no LLM in the execution path)
  - State (partial tasks, history) is stored in Redis, never in LLM

Flow:
  handle_message()
    ├─ Pending state? → complete partial task (datetime) or yes/no
    ├─ Route intent → action handler
    │     add    → if time present: save directly
    │              if time missing: ask when, store partial intent
    │     list   → Notion date query → numbered output
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
    get_ctx,
    get_last_task,
    get_pending_task,
    get_user_memos,
    get_user_timezone,
    increment_chat_streak,
    log_behavior,
    record_behavior_pattern,
    reset_chat_streak,
    set_ctx,
    set_last_task,
    set_pending_task,
)
from engine.router import route
from services.ai_service import general_chat, synthesize_response
from services.notion_service import (
    create_task,
    delete_task,
    find_task_by_title,
    get_task_by_id,
    list_tasks,
    list_tasks_by_date_range,
    mark_done,
    next_recurrence_date,
    search_tasks_multi,
    task_to_text,
    update_task,
    _get_rich_text,
    _get_select,
    PROP_RECURRENCE,
    PROP_DUE,
    PROP_PRIORITY,
    PROP_TITLE,
    _CATEGORY_ICONS,
)
from utils.datetime_parser import extract_datetime, format_local

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.55

_YES_WORDS = {"yes", "y", "yep", "yeah", "sure", "ok", "okay", "confirm", "go", "do it", "save", "add it"}
_NO_WORDS  = {"no", "n", "nope", "cancel", "stop", "abort", "nevermind", "never mind", "don't", "dont"}

# Pronouns that mean "the last task I mentioned" — resolved from Redis
_PRONOUNS = {"it", "that", "this", "that one", "this one", "that task", "this task", "the task"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def handle_message(text: str, user_id: str, history: Optional[list] = None) -> str:
    user_tz = await get_user_timezone(user_id)

    # ── 1. Handle pending state ──────────────────────────────────────────────
    pending = await get_pending_task(user_id)
    if pending:
        lowered     = text.strip().lower()
        waiting_for = pending.get("waiting_for")

        if waiting_for == "datetime":
            # User is providing time for a partial task
            if lowered in _NO_WORDS:
                await clear_pending_task(user_id)
                return "Cancelled."

            # They said yes to a suggested time
            suggested = pending.get("suggested_time")
            if suggested and lowered in _YES_WORDS:
                due = extract_datetime(f"at {suggested} tomorrow", user_tz=user_tz)
                if due:
                    await clear_pending_task(user_id)
                    await reset_chat_streak(user_id)
                    return await _execute_task_directly(
                        title=pending["title"], due=due,
                        priority=pending.get("priority", "medium"),
                        category=pending.get("category"),
                        notes=pending.get("notes"),
                        recurrence=pending.get("recurrence", "none"),
                        user_id=user_id, user_tz=user_tz,
                    )

            # Try to parse reply as a datetime
            due = extract_datetime(text, user_tz=user_tz)
            if due:
                await clear_pending_task(user_id)
                await reset_chat_streak(user_id)
                return await _execute_task_directly(
                    title=pending["title"], due=due,
                    priority=pending.get("priority", "medium"),
                    category=pending.get("category"),
                    notes=pending.get("notes"),
                    recurrence=pending.get("recurrence", "none"),
                    user_id=user_id, user_tz=user_tz,
                )
            # Couldn't parse — treat as new message
            await clear_pending_task(user_id)

        else:
            # Legacy yes/no confirmation path
            if lowered in _YES_WORDS:
                return await _execute_confirmed_task(pending, user_id, user_tz)
            if lowered in _NO_WORDS:
                await clear_pending_task(user_id)
                return "Cancelled. What else?"
            await clear_pending_task(user_id)

    # ── 2. Route and dispatch ────────────────────────────────────────────────
    try:
        intent = await route(text, history=history)

        if intent is None or intent.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            return await _chat_fallback(text, user_id, history)

        action = intent["action"]

        if action == "add":
            result = await _add(intent, user_id, user_tz, history)
        elif action == "list":
            result = await _list(intent, user_tz, user_id=user_id)
        elif action == "search":
            # Record the search subject so "delete it" / "done with it" works after
            search_title = (intent.get("title") or intent.get("keyword") or "").strip()
            if search_title:
                await set_last_task(user_id, search_title)
            result = await _search(intent, original_text=text, history=history, user_tz=user_tz)
        elif action == "update":
            result = await _update(intent, user_id, user_tz)
        elif action == "delete":
            result = await _delete(intent, user_id)
        elif action == "done":
            result = await _done(intent, user_id, user_tz)
        else:
            return await _chat_fallback(text, user_id, history)

        # Any resolved task action resets the chat drift counter
        await reset_chat_streak(user_id)
        return result

    except Exception as exc:
        logger.exception("handle_message error for '%s': %s", text, exc)
        return await general_chat(text, history=history)


# ---------------------------------------------------------------------------
# ADD — save directly if time present; ask for time if not
# ---------------------------------------------------------------------------
async def _add(intent: dict, user_id: str, user_tz: str, history: Optional[list]) -> str:
    title = (intent.get("title") or "").strip()
    if not title:
        return "What should I call this task?"

    due        = intent.get("datetime")
    priority   = intent.get("priority", "medium")
    category   = intent.get("category")
    notes      = intent.get("notes")
    recurrence = intent.get("recurrence") or "none"

    # ── No time → ask once, store partial intent ─────────────────────────────
    if not due:
        suggestion = await _suggest_time(user_id, category)
        task_data  = {
            "title": title, "priority": priority,
            "category": category, "notes": notes, "recurrence": recurrence,
            "waiting_for": "datetime",
        }
        if suggestion:
            task_data["suggested_time"] = suggestion
            cat_str = f"{category} tasks" if category else "tasks"
            await set_pending_task(user_id, task_data)
            return (
                f"When should I schedule *{title}*?\n"
                f"_(You usually do {cat_str} at {suggestion} — reply *yes* to use that)_"
            )
        await set_pending_task(user_id, task_data)
        return f"When should I schedule *{title}*? _(e.g. tomorrow at 5pm)_"

    # ── Validate datetime ────────────────────────────────────────────────────
    try:
        dt = datetime.fromisoformat(due)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt <= datetime.now(timezone.utc):
            return f"That time is in the past. When should I schedule *{title}*?"
    except ValueError:
        return "Couldn't parse that date/time. Try _tomorrow at 5pm_."

    # ── Time is clear → save directly, no confirmation needed ───────────────
    return await _execute_task_directly(
        title=title, due=due, priority=priority,
        category=category, notes=notes, recurrence=recurrence,
        user_id=user_id, user_tz=user_tz,
    )


async def _execute_task_directly(
    title: str,
    due: str,
    priority: str,
    category: Optional[str],
    notes: Optional[str],
    recurrence: str,
    user_id: str,
    user_tz: str,
) -> str:
    """Save task to Notion immediately without a confirmation step."""
    await create_task(
        title, due,
        priority=priority,
        category=category,
        notes=notes,
        recurrence=recurrence if recurrence != "none" else None,
    )
    await log_behavior(user_id, "created", title)

    if due:
        try:
            dt = datetime.fromisoformat(due).astimezone(timezone.utc)
            await record_behavior_pattern(user_id, category, dt.hour)
        except Exception:
            pass

    local_time = format_local(due, user_tz) if due else "No time set"
    cat_icon   = _CATEGORY_ICONS.get(category or "", "")
    icon_part  = f" {cat_icon}" if cat_icon else ""
    recur_part = " ♻️" if recurrence and recurrence != "none" else ""
    note_part  = f"\n  ↳ _{notes}_" if notes else ""

    return f"✅ *{title}*{icon_part} — {local_time}{recur_part}{note_part}"


async def _execute_confirmed_task(pending: dict, user_id: str, user_tz: str) -> str:
    """Legacy: execute a task that went through the old yes/no confirmation flow."""
    await clear_pending_task(user_id)
    return await _execute_task_directly(
        title=pending["title"],
        due=pending.get("due", ""),
        priority=pending.get("priority", "medium"),
        category=pending.get("category"),
        notes=pending.get("notes"),
        recurrence=pending.get("recurrence", "none"),
        user_id=user_id,
        user_tz=user_tz,
    )


# ---------------------------------------------------------------------------
# LIST — numbered, concise
# ---------------------------------------------------------------------------
async def _list(intent: dict, user_tz: str, user_id: Optional[str] = None) -> str:
    from datetime import timedelta
    date_range   = (intent.get("date_range") or "today").lower()
    include_done = intent.get("include_done", False)
    now          = datetime.now(timezone.utc)

    if date_range == "today":
        tasks = await list_tasks(filter_date=now.strftime("%Y-%m-%d"), include_done=include_done)
        label = "Today"
    elif date_range == "tomorrow":
        tasks = await list_tasks(
            filter_date=(now + timedelta(days=1)).strftime("%Y-%m-%d"),
            include_done=include_done,
        )
        label = "Tomorrow"
    elif date_range in ("last_7_days", "last_15_days", "last_30_days"):
        days  = int(date_range.split("_")[1])
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = now.strftime("%Y-%m-%d")
        tasks = await list_tasks_by_date_range(start, end, include_done=True)
        label = f"Last {days} days"
    elif date_range == "this_week":
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
        empty = {
            "Today":    "Nothing scheduled for today 🎉",
            "Tomorrow": "Nothing scheduled for tomorrow yet.",
        }
        return empty.get(label, f"No tasks found for *{label}*.")

    # Save to context so "delete 1" / "done 2" resolves correctly
    if user_id:
        await set_ctx(user_id, last_task_list=[
            {"id": p["id"], "title": _get_title_from_page(p)} for p in tasks
        ], last_action="list")

    lines = [f"📋 *{label}:*"]
    for i, p in enumerate(tasks, 1):
        lines.append(f"{i}. {task_to_text(p, user_tz)}")
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

    tasks      = await search_tasks_multi(
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
        ctx = await get_ctx(user_id)
        if not ctx.get("last_task_title") and not ctx.get("last_task_list"):
            return "Which task should I update?"
    if not update_field or not update_value:
        return "What should I change? e.g. _move to 6pm_ or _mark as high priority_"

    page, actual_title = await _resolve_task_ref(title, user_id)
    if not page:
        return await _not_found_msg(title or "that task", user_id)
    page_id = page["id"]
    kwargs: dict = {}

    if update_field == "datetime":
        new_dt = extract_datetime(update_value, user_tz=user_tz)
        if not new_dt:
            return f"Couldn't parse '{update_value}'. Try _tomorrow at 6pm_."
        kwargs["due_iso"] = new_dt
        msg = f"📅 *{actual_title}* rescheduled to {format_local(new_dt, user_tz)}"

    elif update_field == "priority":
        p = update_value.lower()
        if p not in {"low", "medium", "high"}:
            return "Priority must be low, medium, or high."
        kwargs["priority"] = p
        msg = f"🔔 *{actual_title}* priority → {p}"

    elif update_field == "title":
        kwargs["title"] = update_value
        msg = f"✏️ Renamed to *{update_value}*"

    elif update_field == "status":
        s = update_value.lower()
        if s in {"done", "complete", "completed", "finished"}:
            kwargs["status"] = "done"
            msg = f"✅ *{actual_title}* marked done"
        else:
            kwargs["status"] = "todo"
            msg = f"🔄 *{actual_title}* marked todo"
    else:
        return "I can update: datetime, priority, title, or status."

    await update_task(page_id, **kwargs)
    await log_behavior(user_id, "updated", actual_title)
    return msg


# ---------------------------------------------------------------------------
# DONE — mark complete + spawn next if recurring
# ---------------------------------------------------------------------------
async def _done(intent: dict, user_id: str, user_tz: str) -> str:
    title  = (intent.get("title") or "").strip()
    page, actual_title = await _resolve_task_ref(title, user_id)

    if not page:
        return await _not_found_msg(title or "that task", user_id)

    page_id    = page["id"]
    props      = page.get("properties", {})
    recurrence = _get_rich_text(props, PROP_RECURRENCE)

    await set_ctx(user_id, last_task_title=actual_title)
    await mark_done(page_id)
    await log_behavior(user_id, "completed", actual_title)

    follow_up = ""
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
                follow_up = f"\n♻️ Next: {next_local}"
        except Exception as exc:
            logger.warning("Recurrence expansion failed: %s", exc)

    return f"✅ Marked done: *{actual_title}*{follow_up}"


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------
async def _delete(intent: dict, user_id: str) -> str:
    title  = (intent.get("title") or "").strip()
    page, actual_title = await _resolve_task_ref(title, user_id)

    if not page:
        return await _not_found_msg(title or "that task", user_id)

    await set_ctx(user_id, last_task_title=actual_title)
    await delete_task(page["id"])
    await log_behavior(user_id, "deleted", actual_title)
    return f"🗑️ Deleted *{actual_title}*"


# ---------------------------------------------------------------------------
# Task reference resolution (number → list position, pronoun → last task, fuzzy)
# ---------------------------------------------------------------------------
async def _resolve_task_ref(ref: str, user_id: str):
    """
    Resolve any task reference to a (page, actual_title) tuple.
    Returns (None, None) when nothing matches.

    Priority order:
      1. Digit → ctx["last_task_list"][n-1]  (number-based control)
      2. Pronoun → ctx["last_task_title"]
      3. Exact title → find_task_by_title()
      4. Fuzzy keyword → search_tasks_multi()
    """
    ref = (ref or "").strip()
    ctx = await get_ctx(user_id)

    # 1. Number reference: "delete 1", "done 2", "move 3 to tomorrow"
    if ref.isdigit():
        n = int(ref)
        task_list = ctx.get("last_task_list", [])
        if task_list and 1 <= n <= len(task_list):
            entry = task_list[n - 1]
            page = await get_task_by_id(entry["id"])
            if not page or page.get("archived"):
                page = await find_task_by_title(entry["title"])
            if page:
                return page, entry["title"]
        return None, None

    # 2. Pronoun: "delete it", "done with that", "move this to tomorrow"
    if not ref or ref.lower() in _PRONOUNS:
        ref = ctx.get("last_task_title", "")
        if not ref:
            return None, None

    # 3. Exact title match
    page = await find_task_by_title(ref)
    if page:
        return page, _get_title_from_page(page)

    # 4. Fuzzy / keyword fallback (min 2-char words)
    words = [w for w in ref.split() if len(w) > 2]
    if words:
        results = await search_tasks_multi(keywords=words)
        if results:
            page = results[0]
            return page, _get_title_from_page(page)

    return None, None


async def _not_found_msg(ref: str, user_id: str) -> str:
    """Return a helpful suggestion instead of a dead-end 'not found' message."""
    ctx = await get_ctx(user_id)
    task_list = ctx.get("last_task_list", [])
    if task_list:
        numbered = "\n".join(
            f"{i + 1}. {t['title']}" for i, t in enumerate(task_list[:4])
        )
        return f"Couldn't find *{ref}*. Last tasks shown:\n{numbered}\n\nReply with the number to act on one."
    # No list context — try a broad search
    results = await list_tasks(include_done=False)
    if results:
        titles = ", ".join(_get_title_from_page(p) for p in results[:3])
        return f"Couldn't find *{ref}*. Active tasks: {titles}"
    return f"Couldn't find *{ref}*. You have no active tasks right now."


# ---------------------------------------------------------------------------
# Behavioral intelligence — suggest times based on past patterns
# ---------------------------------------------------------------------------
async def _suggest_time(user_id: str, category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    try:
        patterns = await get_behavior_patterns(user_id)
        hours    = patterns.get(f"hours_{category}", {})
        if not hours:
            return None
        best_hour = max(hours, key=lambda h: hours[h])
        h = int(best_hour)
        if h == 0:   return "12am"
        if h < 12:   return f"{h}am"
        if h == 12:  return "12pm"
        return f"{h - 12}pm"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Chat fallback — task-aware, streak-limited
# ---------------------------------------------------------------------------
async def _chat_fallback(text: str, user_id: str, history: Optional[list]) -> str:
    """Route to general_chat with injected task context and drift streak."""
    task_context  = await _get_task_context()
    user_profile  = await _build_user_profile(user_id)
    streak        = await increment_chat_streak(user_id)
    return await general_chat(
        text,
        history=history,
        task_context=task_context,
        user_profile=user_profile,
        chat_streak=streak,
    )


async def _get_task_context() -> Optional[str]:
    """Return a one-line summary of today's tasks for chat context injection."""
    try:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tasks = await list_tasks(filter_date=today, include_done=False)
        if not tasks:
            return "no tasks today"
        count = len(tasks)
        high  = sum(
            1 for t in tasks
            if _get_select(t.get("properties", {}), PROP_PRIORITY) == "high"
        )
        s = f"{count} task{'s' if count != 1 else ''} today"
        if high:
            s += f" ({high} high priority)"
        return s
    except Exception:
        return None


async def _build_user_profile(user_id: str) -> Optional[str]:
    """Derive a brief user profile string from patterns and memos."""
    try:
        patterns = await get_behavior_patterns(user_id)
        memos    = await get_user_memos(user_id)
        signals: list[str] = []

        # Top category
        categories = patterns.get("categories", {})
        if categories:
            top = max(categories, key=lambda c: categories[c])
            signals.append(f"mostly {top} tasks")

        # Preferred time-of-day
        all_hours: dict[int, int] = {}
        for key, val in patterns.items():
            if key.startswith("hours_") and isinstance(val, dict):
                for h, n in val.items():
                    all_hours[int(h)] = all_hours.get(int(h), 0) + n
        if all_hours:
            best = max(all_hours, key=lambda h: all_hours[h])
            period = "morning" if best < 12 else ("afternoon" if best < 17 else "evening")
            signals.append(f"prefers {period}")

        # Free-form memos
        signals.extend(memos.values())

        return "; ".join(signals) if signals else None
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
