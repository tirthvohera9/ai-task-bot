"""
FastAPI entry point — Vercel serverless edition.

Each request is stateless:
  - /webhook      → creates a fresh bot Application, processes the update, tears down
  - /cron/reminders → checks Notion for due tasks, sends Telegram reminders
  - /cron/summary   → sends the 08:00 UTC daily summary
  - /setup          → registers the Telegram webhook (run once after deploy)
  - /health         → liveness check
"""
import logging

from fastapi import FastAPI, Header, Request, Response
from telegram import Update
from telegram.ext import Application

from config import settings
from engine.scheduler import send_daily_summary, send_reminders
from routes.telegram import setup_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Task Manager")


# ---------------------------------------------------------------------------
# Telegram webhook
# ---------------------------------------------------------------------------
@app.post("/webhook")
async def webhook(request: Request) -> Response:
    """Receive and process a Telegram update."""
    data = await request.json()

    bot_app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    setup_handlers(bot_app)
    await bot_app.initialize()

    try:
        update = Update.de_json(data, bot_app.bot)
        await bot_app.process_update(update)
    finally:
        await bot_app.shutdown()

    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Cron endpoints  (called by Vercel on schedule — secured by CRON_SECRET)
# ---------------------------------------------------------------------------
def _verify_cron(authorization: str | None) -> bool:
    return authorization == f"Bearer {settings.CRON_SECRET}"


@app.get("/cron/reminders")
async def cron_reminders(authorization: str | None = Header(default=None)) -> dict:
    """Every-minute cron: send reminders for tasks due soon."""
    if not _verify_cron(authorization):
        return Response(status_code=401)  # type: ignore[return-value]

    sent = await send_reminders()
    return {"ok": True, "reminders_sent": sent}


@app.get("/cron/summary")
async def cron_summary(authorization: str | None = Header(default=None)) -> dict:
    """Daily 08:00 UTC cron: send today's task summary."""
    if not _verify_cron(authorization):
        return Response(status_code=401)  # type: ignore[return-value]

    await send_daily_summary()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Setup — call once after deploying to register the Telegram webhook
# ---------------------------------------------------------------------------
@app.get("/setup")
async def setup(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    """
    Register the Telegram webhook URL.
    Secured with CRON_SECRET header.
    Uses the incoming request's host as the webhook base URL
    so it always points to the production domain.
    """
    if not _verify_cron(authorization):
        return Response(status_code=401)  # type: ignore[return-value]

    # Build webhook URL from the actual request host (always correct)
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    webhook_url = f"https://{host}/webhook"

    bot_app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()
    await bot_app.initialize()
    await bot_app.bot.set_webhook(url=webhook_url)
    await bot_app.shutdown()

    logger.info("Webhook registered: %s", webhook_url)
    return {"ok": True, "webhook_url": webhook_url}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
