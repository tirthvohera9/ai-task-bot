"""
Voice-to-text via Groq Whisper (free tier).
Groq offers whisper-large-v3 free at 7,200 audio seconds/day.
API is OpenAI-compatible — no extra dependency needed.
"""
import logging
import tempfile
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

# Groq uses the OpenAI-compatible endpoint
_client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


async def transcribe_voice(file_id: str) -> str:
    """
    1. Resolve file_id → download URL via Telegram getFile.
    2. Download the .ogg audio.
    3. Send to Whisper API.
    4. Return transcribed text.
    """
    download_url = await _get_download_url(file_id)
    audio_bytes = await _download(download_url)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as f:
            response = await _client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",  # Groq free tier model
                file=f,
                response_format="text",
            )
        text = response.strip() if isinstance(response, str) else response.text.strip()
        logger.info("Whisper transcription: %s", text)
        return text
    finally:
        tmp_path.unlink(missing_ok=True)


async def _get_download_url(file_id: str) -> str:
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/getFile"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params={"file_id": file_id})
        resp.raise_for_status()
        file_path = resp.json()["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{settings.TELEGRAM_BOT_TOKEN}/{file_path}"


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content
