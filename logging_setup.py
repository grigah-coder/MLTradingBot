"""Centralised logging configuration.

Sets up a single root configuration with:

* a console handler (stdout), and
* a size-bounded **rotating file handler** so logs survive restarts without
  growing without bound.

Both handlers share one formatter that includes a timestamp, level, logger
name and thread name (important now that symbols are analysed concurrently).

Usage::

    from logging_setup import configure_logging, get_logger

    configure_logging()          # call once, early (idempotent)
    logger = get_logger(__name__)
    logger.info("hello")
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional

_configured = False

# Third-party libraries that are extremely chatty at DEBUG/INFO.
_NOISY_LOGGERS = (
    "urllib3",
    "requests",
    "httpx",
    "httpcore",
    "asyncio",
    "apscheduler",
    "transformers",
    "huggingface_hub",
    "filelock",
    "matplotlib",
    "PIL",
    "yfinance",
)

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _coerce_level(level) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def configure_logging(
    level=None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    quiet_libs: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Idempotently configure the root logger. Returns the root logger."""
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return root

    # Import lazily to avoid a circular import with config.
    try:
        from config import (  # type: ignore
            LOG_LEVEL,
            LOG_DIR,
            LOG_FILE,
            LOG_MAX_BYTES,
            LOG_BACKUP_COUNT,
        )

        resolved_level = _coerce_level(level if level is not None else LOG_LEVEL)
        log_dir = log_dir or LOG_DIR
        log_file = log_file or LOG_FILE
        max_bytes = max_bytes if max_bytes is not None else LOG_MAX_BYTES
        backup_count = backup_count if backup_count is not None else LOG_BACKUP_COUNT
    except Exception:  # pragma: no cover - config should always exist
        resolved_level = _coerce_level(level if level is not None else "INFO")
        log_dir = log_dir or "logs"
        log_file = log_file or "tradingbot.log"
        max_bytes = max_bytes if max_bytes is not None else 5_000_000
        backup_count = backup_count if backup_count is not None else 5

    # Reset any previously-attached handlers (e.g. when re-configuring).
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    root.setLevel(resolved_level)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(resolved_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_file),
            maxBytes=int(max_bytes),
            backupCount=int(backup_count),
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        root.warning("File logging disabled (%s)", exc)

    if quiet_libs:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger (``mltrader`` by default)."""
    return logging.getLogger(name if name else "mltrader")
