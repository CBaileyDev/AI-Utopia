"""Structured logging setup. Default: human-readable to stderr; JSON when AIUTOPIA_LOG_JSON=1."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      time.time(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k in {"args", "msg", "exc_info", "exc_text", "stack_info",
                     "lineno", "pathname", "filename", "funcName",
                     "module", "msecs", "relativeCreated", "thread",
                     "threadName", "processName", "process", "name",
                     "levelname", "levelno", "created"}:
                continue
            payload[k] = v
        return json.dumps(payload, default=str)


def setup_logging(level: str | int = "INFO") -> None:
    """Configure the root logger. Call once at process start."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("AIUTOPIA_LOG_JSON") == "1":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
