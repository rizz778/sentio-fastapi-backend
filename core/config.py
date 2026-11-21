import json
from pathlib import Path
from typing import List, Optional
from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=SERVER_DIR / ".env",
        extra="ignore",
    )

    APP_NAME: str = "FastAPI Backend"
    ENV: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    # gRPC Settings
    GRPC_HOST: str = "0.0.0.0"
    GRPC_PORT: int = 50051

    # Redis & Upstream Service Settings
    REDIS_URL: str = "redis://localhost:6379"
    NODE_BACKEND_URL: str = "http://localhost:5000"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4000",
        "http://127.0.0.1:4000",
        "*",
    ]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # options: json, console
    LOG_FILE_PATH: Optional[Path] = None
    ENABLE_FILE_LOGGING: bool = False
    ENABLE_CONSOLE_LOGGING: bool = True

    # Gemini Models, Voice & Keys
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_VERSION: str = "v1beta"
    GEMINI_MODEL: Optional[str] = None
    GEMINI_TEXT_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_LIVE_TEXT_MODEL: str = "gemini-3.8-live"
    GEMINI_AUDIO_MODEL: str = "gemini-3.8-live"
    GEMINI_DISTRESS_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_VISION_MODEL: str = "gemini-3.8-flash"
    GEMINI_VOICE_NAME: str = "Puck"
    DEFAULT_SYSTEM_INSTRUCTION: str = "You are a Therapist"
    DISTRESS_API_KEY: Optional[str] = None

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        if isinstance(v, str) and v.lower() in {"release", "production", "prod"}:
            return False
        return v

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("LOG_FILE_PATH", mode="before")
    @classmethod
    def validate_log_path(cls, v, info: ValidationInfo):
        if info.data.get("ENABLE_FILE_LOGGING", False):
            if not v:
                raise ValueError("LOG_FILE_PATH must be set when ENABLE_FILE_LOGGING is True")
            path = Path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return v


settings = Settings()
