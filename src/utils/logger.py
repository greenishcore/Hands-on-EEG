"""
Unified logging configuration for the Hands-on-EEG project.

Usage
-----
>>> from src.utils.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("Loading data …")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    """Create (or retrieve) a named logger with console and optional file output.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    level:
        Log level string – one of ``DEBUG``, ``INFO``, ``WARNING``,
        ``ERROR``, or ``CRITICAL``.
    log_file:
        Optional path to a log file.  If *None* only the console handler
        is attached.
    fmt:
        Log record format string passed to :class:`logging.Formatter`.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the logger is retrieved again.
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_from_config(cfg: object) -> None:  # noqa: ANN001
    """Apply logging settings from a :class:`~src.config.Config` object.

    Parameters
    ----------
    cfg:
        A :class:`~src.config.Config` instance whose ``logging`` attribute
        exposes ``level``, ``log_file``, and ``format`` fields.
    """
    log_cfg = getattr(cfg, "logging", None)
    if log_cfg is None:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))
    formatter = logging.Formatter(log_cfg.format)
    for handler in root.handlers:
        handler.setFormatter(formatter)
