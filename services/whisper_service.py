"""
Voice-to-text via Groq Whisper (free tier).
Groq offers whisper-large-v3-turbo free at 7,200 audio seconds/day.
API is OpenAI-compatible — no extra dependency needed.

Returns (transcribed_text, confidence_score).
Confidence is derived from segment avg_logprob (0.0–1.0 scale).
"""
import logging
import math
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


async def transcribe_voice(file_id: str) -> tuple[str, float]:
    """
    Transcribe a Telegram voice file.
    Returns (text, confidence) where confidence is 0.0–1.0.
    """
    download_url = await _get_download_url(file_id)
    audio_bytes  = await _download(download_url)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as f:
            response = await _client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="verbose_json",   # gives us segments + logprob
            )

        # Extract text
        if isinstance(response, str):
            text       = response.strip()
            confidence = 0.85   # no logprob available, assume decent
        else:
            text       = (getattr(response, "text", None) or "").strip()
            confidence = _compute_confidence(response)

        logger.info("Whisper: '%s' (confidence=%.2f)", text, confidence)
        return text, confidence

    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


def _compute_confidence(response) -> float:
    """
    Convert Whisper segment avg_logprob → a 0–1 confidence score.
    avg_logprob is negative; closer to 0 = more confident.
    Typical range: -0.2 (excellent) to -1.5 (poor).
    """
    try:
        segments = getattr(response, "segments", None) or []
        if not segments:
            return 0.85

        avg_logprob = sum(
            getattr(s, "avg_logprob", -0.5) for s in segments
        ) / len(segments)

        # Map [-2.0, 0.0] → [0.0, 1.0]
        score = max(0.0, min(1.0, 1.0 + avg_logprob / 2.0))
        return round(score, 3)
    except Exception:
        return 0.85


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
