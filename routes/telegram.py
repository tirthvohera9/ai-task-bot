"""
Telegram message handlers.
Handles: /start, /help, /summary, /clear, plain text, and voice messages.
"""
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from db.database import add_to_history, clear_history, get_history
from engine.task_engine import handle_message
from services.whisper_service import transcribe_voice

logger = logging.getLogger(__name__)


def setup_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start", _start))
    app.add_handler(CommandHandler("help", _help))
    app.add_handler(CommandHandler("summary", _summary))
    app.add_handler(CommandHandler("clear", _clear))
    app.add_handler(MessageHandler(filters.VOICE, _voice_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _text_handler))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hi! I'm your task manager.\n\n"
        "Try:\n"
        "• _Add a meeting tomorrow at 3pm_\n"
        "• _Show today's tasks_\n"
        "• _When do I have to buy milk?_\n"
        "• _What are my reminders for last 7 days?_\n"
        "• _Done with report_\n"
        "• Or send a voice note!\n\n"
        "Use /help for all commands.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*Commands:*\n"
        "/start — Welcome message\n"
        "/help  — This message\n"
        "/summary — Today's tasks\n"
        "/clear — Reset conversation memory\n\n"
        "*Natural language:*\n"
        "• Add [task] at [time]\n"
        "• Remind me to [task] tomorrow\n"
        "• Show today's tasks / last 15 days\n"
        "• When do I have to buy milk?\n"
        "• Done with [task]\n"
        "• Delete [task]\n\n"
        "🎤 Voice messages also work!",
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
        "🧹 Conversation memory cleared! Starting fresh.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    text = update.message.text.strip()

    if not text:
        return

    await update.message.chat.send_action("typing")

    # Load conversation history for context
    history = await get_history(user_id)

    try:
        response = await handle_message(text, user_id, history=history)
    except Exception as exc:
        logger.exception("Unhandled error in text_handler: %s", exc)
        response = "Something went wrong. Please try again."

    # Save this turn to history
    await add_to_history(user_id, "user", text)
    await add_to_history(user_id, "assistant", response)

    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)


async def _voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    voice = update.message.voice

    await update.message.chat.send_action("typing")

    try:
        transcribed = await transcribe_voice(voice.file_id)
    except Exception as exc:
        logger.exception("Whisper transcription failed: %s", exc)
        await update.message.reply_text("❌ Could not transcribe voice message. Please try text.")
        return

    # Echo back what was heard
    await update.message.reply_text(
        f'🎤 _Heard: "{transcribed}"_', parse_mode=ParseMode.MARKDOWN
    )

    # Load conversation history for context
    history = await get_history(user_id)

    try:
        response = await handle_message(transcribed, user_id, history=history)
    except Exception as exc:
        logger.exception("Unhandled error processing voice: %s", exc)
        response = "Something went wrong. Please try again."

    # Save this turn to history
    await add_to_history(user_id, "user", transcribed)
    await add_to_history(user_id, "assistant", response)

    await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
