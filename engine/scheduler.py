"""
Background scheduler: checks for due tasks every SCHEDULER_INTERVAL seconds
and sends Telegram reminders. Deduplicates via SQLite.
"""
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot

from config import settings
from db.database import is_reminder_sent, mark_reminder_sent
from services.notion_service import get_due_soon, list_tasks, task_to_text

logger = logging.getLogger(__name__)


def start_scheduler(bot: Bot) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        _send_reminders,
        "interval",
        seconds=settings.SCHEDULER_INTERVAL,
        args=[bot],
        id="reminders",
        max_instances=1,
    )

    scheduler.add_job(
        _send_daily_summary,
        "cron",
        hour=8,
        minute=0,
        args=[bot],
        id="daily_summary",
        max_instances=1,
    )

    logger.info(
        "Scheduler started (interval=%ss, daily summary at 08:00 UTC)",
        settings.SCHEDULER_INTERVAL,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
async def _send_reminders(bot: Bot) -> None:
    try:
        due_tasks = await get_due_soon(within_minutes=settings.REMINDER_LEAD_MINUTES)
    except Exception as exc:
        logger.error("Scheduler: failed to fetch due tasks: %s", exc)
        return

    for page in due_tasks:
        page_id = page["id"]
        if await is_reminder_sent(page_id):
            continue

        text = task_to_text(page)
        message = f"⏰ *Reminder:*\n{text}"

        try:
            await bot.send_message(
                chat_id=settings.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode="Markdown",
            )
            await mark_reminder_sent(page_id)
            logger.info("Reminder sent for page %s", page_id)
        except Exception as exc:
            logger.error("Failed to send reminder for %s: %s", page_id, exc)


async def _send_daily_summary(bot: Bot) -> None:
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tasks = await list_tasks(filter_date=today)
    except Exception as exc:
        logger.error("Scheduler: failed to fetch today's tasks: %s", exc)
        return

    if not tasks:
        message = "☀️ Good morning! No tasks scheduled for today."
    else:
        lines = ["☀️ *Good morning! Today's tasks:*"]
        for page in tasks:
            lines.append(task_to_text(page))
        message = "\n".join(lines)

    try:
        await bot.send_message(
            chat_id=settings.TELEGRAM_CHAT_ID,
            text=message,
            parse_mode="Markdown",
        )
        logger.info("Daily summary sent (%d tasks)", len(tasks))
    except Exception as exc:
        logger.error("Failed to send daily summary: %s", exc)
