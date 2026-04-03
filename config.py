from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Telegram
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    # OpenRouter — brain for everything
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"

    # Groq — free Whisper transcription (7,200 seconds/day free tier)
    GROQ_API_KEY: str

    # Notion
    NOTION_API_KEY: str
    NOTION_PARENT_PAGE_ID: str

    # App
    WEBHOOK_URL: str = ""
    PORT: int = 8000
    SCHEDULER_INTERVAL: int = 60
    REMINDER_LEAD_MINUTES: int = 15


settings = Settings()
