"""
Structured logging utilities for Agent Platform.

This module provides a comprehensive logging system with support for:
- JSON formatted logs for production
- Pretty colored logs for development
- Request ID tracking for distributed tracing
- Rotating file handlers
- Configurable log levels
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

from agent_platform.config import get_settings

# Context variable for request ID tracking
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in production environments.

    Formats log records as JSON objects with timestamp, level, logger name,
    message, and any extra fields including request_id if present.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON formatted string of the log record.
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if present
        request_id = _request_id.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add filename and line number for DEBUG level
        if record.levelno == logging.DEBUG:
            log_data["location"] = f"{record.filename}:{record.lineno}"

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for development environments.

    Uses Rich library for colorized and formatted console output.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors and formatting.

        Args:
            record: The log record to format.

        Returns:
            Formatted string of the log record.
        """
        # Add request ID to message if present
        request_id = _request_id.get()
        if request_id:
            record.msg = f"[{request_id}] {record.msg}"

        # Add location for DEBUG level
        if record.levelno == logging.DEBUG:
            record.msg = f"{record.msg} ({record.filename}:{record.lineno})"

        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    log_file: str = "agent_platform.log",
) -> None:
    """
    Set up the logging configuration for the application.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, uses the level from settings.
        log_dir: Directory for log files. If None, uses 'logs' in current directory.
        log_file: Name of the log file. Defaults to 'agent_platform.log'.
    """
    settings = get_settings()
    level = log_level or settings.log_level

    # Create log directory if it doesn't exist
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with appropriate formatter
    if settings.is_development():
        # Use Rich handler for pretty colored logs in development
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setFormatter(
            logging.Formatter("%(message)s", datefmt="[%X]")
        )
    else:
        # Use JSON formatter for production
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())

    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # Rotating file handler with JSON formatter
    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    # Log initial setup message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized",
        extra={
            "level": level,
            "environment": settings.environment,
            "log_file": str(log_dir / log_file),
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    This function returns a logger with the specified name. The logger
    inherits the configuration from the root logger set up by setup_logging().

    Args:
        name: The name of the logger, typically __name__ of the calling module.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Agent started", extra={"model": "gpt-4"})
    """
    return logging.getLogger(name)


def set_request_id(request_id: str) -> None:
    """
    Set the request ID for the current context.

    This allows tracking of related log messages across async operations
    and different parts of the application.

    Args:
        request_id: The unique request identifier.

    Example:
        >>> set_request_id("req-123-456-789")
        >>> logger.info("Processing request")  # Will include request_id
    """
    _request_id.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Get the current request ID from context.

    Returns:
        The current request ID, or None if not set.

    Example:
        >>> set_request_id("req-123")
        >>> print(get_request_id())
        'req-123'
    """
    return _request_id.get()


def clear_request_id() -> None:
    """
    Clear the request ID from the current context.

    This should be called after request processing is complete.

    Example:
        >>> set_request_id("req-123")
        >>> # ... process request ...
        >>> clear_request_id()
    """
    _request_id.set(None)


# Initialize logging on module import
try:
    setup_logging()
except Exception as e:
    # Fallback to basic logging if setup fails
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger(__name__).warning(
        f"Failed to set up advanced logging: {e}. Using basic logging."
    )
