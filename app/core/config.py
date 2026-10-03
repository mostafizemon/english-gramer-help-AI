from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "English Grammar AI API"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False

    # Gemini
    GEMINI_API_KEYS: str = Field(default="", repr=False)
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # Groq
    GROQ_API_KEYS: str = Field(default="", repr=False)
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # AI
    AI_REQUEST_TIMEOUT_SECONDS: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @staticmethod
    def _parse_keys(keys_raw: str) -> list[str]:
        if not keys_raw:
            return []

        return [
            key.strip()
            for key in keys_raw.split(",")
            if key.strip()
        ]

    @property
    def gemini_api_keys(self) -> list[str]:
        return self._parse_keys(self.GEMINI_API_KEYS)

    @property
    def groq_api_keys(self) -> list[str]:
        return self._parse_keys(self.GROQ_API_KEYS)


settings = Settings()