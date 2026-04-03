"""
Scheduler functions — called by the Vercel Cron Job endpoint every minute.
No APScheduler needed; Vercel handles the scheduling via vercel.json.
"""
import logging
from datetime import datetime, timezone

from telegram import Bot

from config import settings
from db.database import is_reminder_sent, mark_reminder_sent
from services.notion_service import get_due_soon, list_tasks, task_to_text

logger = logging.getLogger(__name__)


async def send_reminders() -> int:
    """
    Check Notion for tasks due within REMINDER_LEAD_MINUTES.
    Send a Telegram message for each unsent reminder.
    Returns the number of reminders sent.
    """
    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    sent = 0

    try:
        due_tasks = await get_due_soon(within_minutes=settings.REMINDER_LEAD_MINUTES)
    except Exception as exc:
        logger.error("Failed to fetch due tasks: %s", exc)
        return 0

    for page in due_tasks:
        page_id = page["id"]

        if await is_reminder_sent(page_id):
            continue

        text = task_to_text(page)
        try:
            await bot.send_message(
                chat_id=settings.TELEGRAM_CHAT_ID,
                text=f"⏰ *Reminder:*\n{text}",
                parse_mode="Markdown",
            )
            await mark_reminder_sent(page_id)
            sent += 1
            logger.info("Reminder sent for page %s", page_id)
        except Exception as exc:
            logger.error("Failed to send reminder for %s: %s", page_id, exc)

    return sent


async def send_daily_summary() -> None:
    """Send today's task list. Called by the 08:00 UTC cron job."""
    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        tasks = await list_tasks(filter_date=today)
    except Exception as exc:
        logger.error("Failed to fetch today's tasks: %s", exc)
        return

    if not tasks:
        message = "☀️ Good morning! No tasks scheduled for today."
    else:
        lines = ["☀️ *Good morning! Today's tasks:*"]
        lines.extend(task_to_text(p) for p in tasks)
        message = "\n".join(lines)

    try:
        await bot.send_message(
            chat_id=settings.TELEGRAM_CHAT_ID,
            text=message,
            parse_mode="Markdown",
        )
    except Exception as exc:
        logger.error("Failed to send daily summary: %s", exc)
