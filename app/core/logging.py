import logging
import logging.config
import sys
from typing import Dict, Any
from pathlib import Path

from app.core.config import settings


def setup_logging() -> None:
    """Set up application logging configuration."""

    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": settings.log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "default",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "detailed",
                "filename": "logs/ml-service.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/ml-service-errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            # FastAPI loggers
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console", "error_file"],
                "level": "ERROR",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            # Application loggers
            "app": {
                "handlers": ["console", "file"],
                "level": settings.log_level,
                "propagate": False,
            },
            "app.ml": {
                "handlers": ["console", "file"],
                "level": settings.log_level,
                "propagate": False,
            },
            "app.api": {
                "handlers": ["console", "file"],
                "level": settings.log_level,
                "propagate": False,
            },
            # Third-party loggers
            "boto3": {
                "handlers": ["console", "file"],
                "level": "WARNING",
                "propagate": False,
            },
            "botocore": {
                "handlers": ["console", "file"],
                "level": "WARNING",
                "propagate": False,
            },
            "s3transfer": {
                "handlers": ["console", "file"],
                "level": "WARNING",
                "propagate": False,
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file", "error_file"],
        },
    }

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Apply logging configuration
    logging.config.dictConfig(logging_config)

    # Set up request ID context (useful for tracing)
    logger = logging.getLogger("app")
    logger.info(f"Logging configured with level: {settings.log_level}")


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records."""

    def filter(self, record):
        # Add request ID if available (would be set by middleware)
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    logger = logging.getLogger(name)
    logger.addFilter(RequestContextFilter())
    return logger


# Application-specific logger instances
app_logger = get_logger("app")
api_logger = get_logger("app.api")
ml_logger = get_logger("app.ml")
