from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Telegram
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    # OpenRouter — brain for everything
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "qwen/qwen3-8b:free"

    # Groq — free Whisper transcription
    GROQ_API_KEY: str

    # Notion
    NOTION_API_KEY: str
    NOTION_PARENT_PAGE_ID: str

    # Upstash Redis — auto-created by Vercel's Upstash integration
    KV_REST_API_URL: str
    KV_REST_API_TOKEN: str

    # Vercel cron job security secret
    CRON_SECRET: str

    # App
    REMINDER_LEAD_MINUTES: int = 15


settings = Settings()
