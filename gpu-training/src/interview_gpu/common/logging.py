"""Logging setup shared by all CLIs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "interview_gpu"
_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def get_logger() -> logging.Logger:
    """Return the project logger."""
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure console (and optionally file) logging.

    Safe to call repeatedly: previous handlers are replaced, so consecutive
    experiments in one process do not write into each other's log files.
    """
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
