"""
Telegram message handlers.

Commands: /start, /help, /summary, /clear, /timezone
Messages: text, voice
"""
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from db.database import (
    add_to_history,
    clear_history,
    get_history,
    get_user_timezone,
    set_user_timezone,
)
from engine.task_engine import handle_message
from services.ai_service import general_chat
from services.whisper_service import transcribe_voice

logger = logging.getLogger(__name__)

# Minimum Whisper confidence to process without asking for clarification
VOICE_CONFIDENCE_THRESHOLD = 0.70


def setup_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start",    _start))
    app.add_handler(CommandHandler("help",     _help))
    app.add_handler(CommandHandler("summary",  _summary))
    app.add_handler(CommandHandler("clear",    _clear))
    app.add_handler(CommandHandler("timezone", _timezone))
    app.add_handler(MessageHandler(filters.VOICE, _voice_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _text_handler))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hi! I'm your AI task manager.\n\n"
        "I understand natural language — just tell me what you need:\n"
        "• _Add dentist appointment next Monday at 3pm_\n"
        "• _I need to buy groceries tomorrow_\n"
        "• _What do I have this week?_\n"
        "• _Move the meeting to 6pm_\n"
        "• _Any health appointments coming up?_\n"
        "• Or send a 🎤 voice note!\n\n"
        "Use /help to see all commands.\n"
        "⚠️ Set your timezone first: /timezone Asia/Kolkata",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    tz      = await get_user_timezone(user_id)
    await update.message.reply_text(
        f"*Commands:*\n"
        f"/start — Welcome\n"
        f"/help — This message\n"
        f"/summary — Today's tasks\n"
        f"/clear — Reset conversation memory\n"
        f"/timezone <Zone> — Set your timezone (current: `{tz}`)\n\n"
        f"*Adding tasks:*\n"
        f"• add meeting tomorrow at 3pm\n"
        f"• remind me to call mom tonight\n"
        f"• I need to pay rent by the 5th\n"
        f"• every Monday gym at 7am _(recurring)_\n\n"
        f"*Editing tasks:*\n"
        f"• move meeting to 6pm\n"
        f"• reschedule dentist to next week\n"
        f"• mark gym as high priority\n\n"
        f"*Finding tasks:*\n"
        f"• what do I have today/this week?\n"
        f"• any doctor appointments?\n"
        f"• when do I have to buy milk?\n"
        f"• what did I complete last week?\n\n"
        f"🎤 Voice messages work for everything!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    history = await get_history(user_id)
    response = await handle_message("list today's tasks", user_id, history=history)
    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)


async def _clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    await clear_history(user_id)
    await update.message.reply_text(
        "🧹 Conversation memory cleared. Starting fresh!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _timezone(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Set the user's timezone.
    Usage: /timezone Asia/Kolkata
    """
    user_id = str(update.effective_user.id)
    args    = context.args or []

    if not args:
        current = await get_user_timezone(user_id)
        await update.message.reply_text(
            f"Your current timezone is `{current}`.\n\n"
            f"To change it, send:\n`/timezone Asia/Kolkata`\n\n"
            f"Common zones:\n"
            f"• `Asia/Kolkata` (IST)\n"
            f"• `America/New_York` (EST)\n"
            f"• `America/Los_Angeles` (PST)\n"
            f"• `Europe/London` (GMT)\n"
            f"• `Asia/Dubai` (GST)\n"
            f"• `Asia/Singapore` (SGT)",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    tz_name = args[0].strip()
    try:
        ZoneInfo(tz_name)  # validate
        await set_user_timezone(user_id, tz_name)
        await update.message.reply_text(
            f"✅ Timezone set to `{tz_name}`.\n"
            f"All times will now be shown and interpreted in your local timezone.",
            parse_mode=ParseMode.MARKDOWN,
        )
    except (ZoneInfoNotFoundError, Exception):
        await update.message.reply_text(
            f"❌ Unknown timezone: `{tz_name}`\n\n"
            f"Use IANA timezone names, e.g.:\n"
            f"`/timezone Asia/Kolkata`",
            parse_mode=ParseMode.MARKDOWN,
        )


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------
async def _text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    text    = update.message.text.strip()
    if not text:
        return

    await update.message.chat.send_action("typing")
    history = await get_history(user_id)

    try:
        response = await handle_message(text, user_id, history=history)
    except Exception as exc:
        logger.exception("text_handler error: %s", exc)
        response = "Something went wrong. Please try again."

    await add_to_history(user_id, "user",      text)
    await add_to_history(user_id, "assistant", response)
    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)


async def _voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    voice   = update.message.voice

    await update.message.chat.send_action("typing")

    try:
        transcribed, confidence = await transcribe_voice(voice.file_id)
    except Exception as exc:
        logger.exception("Whisper transcription failed: %s", exc)
        await update.message.reply_text("❌ Could not transcribe voice message. Please try text.")
        return

    # Low confidence: show what we heard and ask for clarification
    if confidence < VOICE_CONFIDENCE_THRESHOLD:
        await update.message.reply_text(
            f'🎤 _Heard: "{transcribed}"_', parse_mode=ParseMode.MARKDOWN
        )
        history = await get_history(user_id)
        from services.ai_service import parse_intent
        now    = datetime.now(timezone.utc).isoformat()
        intent = await parse_intent(transcribed, current_time=now, history=history)

        if intent and intent.get("action") == "add" and intent.get("title"):
            await update.message.reply_text(
                f"⚠️ Low confidence. Did you mean: *{intent['title']}*?\n\n"
                f"Reply *yes* to confirm or type the correct task.",
                parse_mode=ParseMode.MARKDOWN,
            )
            from db.database import set_pending_task
            await set_pending_task(user_id, {
                "title":    intent.get("title", ""),
                "due":      intent.get("datetime"),
                "priority": intent.get("priority", "medium"),
                "category": intent.get("category"),
                "notes":    intent.get("notes"),
            })
        else:
            await update.message.reply_text(
                "⚠️ Didn't catch that clearly. Could you repeat or type it?",
                parse_mode=ParseMode.MARKDOWN,
            )
        return

    # High confidence — process silently (no transcript echo)
    history = await get_history(user_id)

    try:
        response = await handle_message(transcribed, user_id, history=history)
    except Exception as exc:
        logger.exception("voice_handler error: %s", exc)
        response = "Something went wrong. Please try again."

    await add_to_history(user_id, "user",      transcribed)
    await add_to_history(user_id, "assistant", response)
    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
