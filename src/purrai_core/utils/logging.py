"""Structured JSON logger."""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger that emits structured JSON to stdout.

    Idempotent: calling with the same name twice returns the same logger
    without adding duplicate handlers.

    Args:
        name: Logger name (used as the ``name`` field in JSON output).
        level: Logging level string, e.g. ``"INFO"``, ``"DEBUG"``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
