"""
Scheduler functions — called by Vercel Cron + cron-job.org every minute.

Reminder priority tiers:
  High   → remind at 30min, 15min, 5min before
  Medium → remind at 15min before (single)
  Low    → remind at 5min before (passive, single)
"""
import logging
from datetime import datetime, timedelta, timezone

from telegram import Bot

from config import settings
from db.database import get_user_timezone, is_reminder_sent, mark_reminder_sent
from services.notion_service import (
    get_due_soon,
    list_tasks,
    task_to_text,
    _get_select,
)

logger = logging.getLogger(__name__)

# Lead times per priority tier (minutes before due)
_REMINDER_SLOTS: dict[str, list[tuple[int, str]]] = {
    "high":   [(30, "30min"), (15, "15min"), (5, "5min")],
    "medium": [(15, "main")],
    "low":    [(5, "passive")],
}


async def send_reminders() -> int:
    """
    Check Notion for upcoming tasks and send Telegram reminders.
    Uses priority-aware multi-slot dedup to avoid double-sending.
    Returns number of reminders sent.
    """
    bot  = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    sent = 0
    now  = datetime.now(timezone.utc)

    # Fetch tasks due within the furthest lead time (30 minutes covers all tiers)
    try:
        due_tasks = await get_due_soon(within_minutes=31)
    except Exception as exc:
        logger.error("Failed to fetch due tasks: %s", exc)
        return 0

    for page in due_tasks:
        page_id  = page["id"]
        props    = page.get("properties", {})
        priority = _get_select(props, "Priority") or "medium"

        # Get due datetime
        due_date = props.get("Due", {}).get("date")
        if not due_date:
            continue
        try:
            due_dt = datetime.fromisoformat(due_date["start"])
            if due_dt.tzinfo is None:
                due_dt = due_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        minutes_until_due = (due_dt - now).total_seconds() / 60
        slots = _REMINDER_SLOTS.get(priority, _REMINDER_SLOTS["medium"])

        for lead_minutes, slot_name in slots:
            # Fire this slot if we're within [lead-1, lead+1] minutes of due time
            if not (lead_minutes - 1 <= minutes_until_due <= lead_minutes + 1):
                continue

            if await is_reminder_sent(page_id, slot=slot_name):
                continue

            label = {
                "30min":   "in 30 minutes",
                "15min":   "in 15 minutes",
                "5min":    "in 5 minutes",
                "main":    "soon",
                "passive": "soon",
            }.get(slot_name, "soon")

            user_tz = await get_user_timezone(settings.TELEGRAM_CHAT_ID)
            text    = task_to_text(page, user_tz)
            try:
                await bot.send_message(
                    chat_id=settings.TELEGRAM_CHAT_ID,
                    text=f"⏰ *Reminder* ({label}):\n{text}",
                    parse_mode="Markdown",
                )
                await mark_reminder_sent(page_id, slot=slot_name)
                sent += 1
                logger.info("Reminder sent [%s|%s] for %s", priority, slot_name, page_id)
            except Exception as exc:
                logger.error("Reminder send failed for %s: %s", page_id, exc)

    return sent


async def send_daily_summary() -> None:
    """Daily 08:00 UTC: send today's task list to the user."""
    bot   = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        tasks = await list_tasks(filter_date=today)
    except Exception as exc:
        logger.error("Failed to fetch daily summary tasks: %s", exc)
        return

    user_tz = await get_user_timezone(settings.TELEGRAM_CHAT_ID)

    if not tasks:
        message = "☀️ Good morning! No tasks scheduled for today."
    else:
        lines = ["☀️ *Good morning! Today's tasks:*"]
        for i, p in enumerate(tasks, 1):
            lines.append(f"{i}. {task_to_text(p, user_tz)}")
        message = "\n".join(lines)

    try:
        await bot.send_message(
            chat_id=settings.TELEGRAM_CHAT_ID,
            text=message,
            parse_mode="Markdown",
        )
    except Exception as exc:
        logger.error("Daily summary send failed: %s", exc)
