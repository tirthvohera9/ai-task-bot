"""
Entry point: FastAPI app with Telegram webhook + APScheduler.

Startup sequence:
  1. Init SQLite
  2. Build python-telegram-bot Application
  3. Register handlers
  4. Start APScheduler (reminders + daily summary)
  5. Register /webhook route with Telegram
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import Application

from config import settings
from db.database import init_db
from engine.scheduler import start_scheduler
from routes.telegram import setup_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_bot_app: Application = None  # global so the webhook route can access it


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bot_app

    # 1. Database
    await init_db()

    # 2. Telegram bot
    _bot_app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    setup_handlers(_bot_app)
    await _bot_app.initialize()
    await _bot_app.start()

    # 3. Scheduler
    scheduler = start_scheduler(_bot_app.bot)
    scheduler.start()

    # 4. Register webhook with Telegram (only if WEBHOOK_URL is configured)
    if settings.WEBHOOK_URL:
        webhook_url = f"{settings.WEBHOOK_URL.rstrip('/')}/webhook"
        await _bot_app.bot.set_webhook(url=webhook_url)
        logger.info("Webhook registered: %s", webhook_url)
    else:
        logger.warning("WEBHOOK_URL not set — Telegram webhook not registered.")

    logger.info("Application started.")
    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    await _bot_app.stop()
    await _bot_app.shutdown()
    logger.info("Application stopped.")


app = FastAPI(title="AI Task Manager", lifespan=lifespan)


@app.post("/webhook")
async def webhook(request: Request) -> Response:
    """Receive Telegram updates."""
    data = await request.json()
    update = Update.de_json(data, _bot_app.bot)
    await _bot_app.process_update(update)
    return Response(status_code=200)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=False,
    )
