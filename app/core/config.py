from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/claimvision_ia"
    MODELS_DIR: str = "models"
    APP_TITLE: str = "ClaimVision IA Service"
    APP_VERSION: str = "3.5.0"
    LOG_LEVEL: str = "INFO"
    ORIGINS: list[str] = ["*"]
    MAX_IMAGE_SIZE_MB: int = 10
    MIN_CONFIDENCE_THRESHOLD: float = 0.35
    OLLAMA_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
