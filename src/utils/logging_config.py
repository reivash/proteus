"""
Centralized Logging Configuration for Proteus

Provides a consistent logging setup across all modules with:
- Console output for interactive sessions
- File logging with rotation
- Structured format for easy parsing
- Separate log files for different components

Usage:
    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Message here")
    logger.warning("Warning message")
    logger.error("Error message", exc_info=True)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default log directory
LOG_DIR = Path(__file__).parent.parent.parent / 'logs'

# Log format
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Configured loggers cache
_loggers = {}


def setup_logging(
    log_dir: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Setup root logging configuration.

    Args:
        log_dir: Directory for log files (default: logs/)
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
    """
    if log_dir is None:
        log_dir = LOG_DIR

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (log_dir / 'scans').mkdir(exist_ok=True)
    (log_dir / 'errors').mkdir(exist_ok=True)
    (log_dir / 'trading').mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # Main file handler (rotating)
    main_log_file = log_dir / f"proteus_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)

    # Error file handler (for ERROR and CRITICAL only)
    error_log_file = log_dir / 'errors' / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    # Ensure logging is setup
    if not logging.getLogger().handlers:
        setup_logging()

    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger


def get_trading_logger() -> logging.Logger:
    """Get logger specifically for trading activity."""
    logger = get_logger('proteus.trading')

    # Add trading-specific file handler if not already present
    log_file = LOG_DIR / 'trading' / f"trading_{datetime.now().strftime('%Y%m%d')}.log"

    # Check if handler already exists
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            if str(log_file) in str(handler.baseFilename):
                return logger

    handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(handler)

    return logger


class LogContext:
    """Context manager for logging with additional context."""

    def __init__(self, logger: logging.Logger, context: str):
        self.logger = logger
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"[{self.context}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(
                f"[{self.context}] Failed after {duration:.2f}s: {exc_val}",
                exc_info=True
            )
        else:
            self.logger.info(f"[{self.context}] Completed in {duration:.2f}s")
        return False


# Convenience function for quick logging
def log_info(message: str, logger_name: str = 'proteus'):
    """Quick info log."""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = 'proteus'):
    """Quick warning log."""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = 'proteus', exc_info: bool = False):
    """Quick error log."""
    get_logger(logger_name).error(message, exc_info=exc_info)


if __name__ == '__main__':
    # Test logging configuration
    setup_logging()

    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test context manager
    with LogContext(logger, "TestOperation"):
        logger.info("Doing some work...")

    # Test trading logger
    trading = get_trading_logger()
    trading.info("Trading activity logged")

    print("\nLogging test complete. Check logs/ directory for output.")
