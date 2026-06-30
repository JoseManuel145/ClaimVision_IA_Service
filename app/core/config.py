from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/claimvision_ia"
    MODELS_DIR: str = "models"
    APP_TITLE: str = "ClaimVision IA Service"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    ORIGINS: list[str] = ["*"]
    MAX_IMAGE_SIZE_MB: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
