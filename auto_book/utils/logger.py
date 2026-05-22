"""Structured logging setup."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: str = "./output/run.log",
) -> logging.Logger:
    """Create and configure the application logger."""

    logger = logging.getLogger("auto_book")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(module)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console)

    if log_to_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the application logger."""

    return logging.getLogger("auto_book")
