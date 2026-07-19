import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging() -> None:
    """Configura el sistema de logging del IA Service."""

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    app_logger = logging.getLogger("ClaimVisionIA")
    app_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    app_logger.addHandler(handler)
    app_logger.propagate = False

    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "ia_service.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    app_logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ClaimVisionIA.{name}")
