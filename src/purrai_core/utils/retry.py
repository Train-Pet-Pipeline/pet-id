"""Tenacity retry decorator with backoff."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from tenacity import retry, stop_after_attempt, wait_fixed

T = TypeVar("T")


def retrying(
    max_attempts: int = 3, wait_seconds: float = 1.5
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a retry decorator with fixed-wait backoff.

    Args:
        max_attempts: Maximum number of attempts before re-raising.
        wait_seconds: Fixed wait duration in seconds between attempts.

    Returns:
        A tenacity ``retry`` decorator configured with the given parameters.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_seconds),
        reraise=True,
    )
