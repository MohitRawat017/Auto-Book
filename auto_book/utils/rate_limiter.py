"""Simple process-local rate limiter for Groq calls."""

import time

from auto_book.config import settings
from auto_book.utils.logger import get_logger

_last_call_time: float = 0.0
_consecutive_failures: int = 0


def wait_for_rate_limit() -> None:
    """Wait between LLM calls to reduce free-tier rate-limit pressure.

    Uses exponential backoff after consecutive failures to avoid
    hammering the API when rate-limited.
    """

    global _last_call_time
    logger = get_logger()
    min_delay = max(0, settings.retry.retry_delay_seconds)

    # Add exponential backoff for consecutive failures
    effective_delay = min_delay + (_consecutive_failures * 3)
    effective_delay = min(effective_delay, 60)  # Cap at 60s

    elapsed = time.time() - _last_call_time
    if elapsed < effective_delay:
        wait_time = effective_delay - elapsed
        logger.debug("Rate limiter sleeping for %.2fs", wait_time)
        time.sleep(wait_time)
    _last_call_time = time.time()


def record_success() -> None:
    """Reset backoff counter after a successful API call."""
    global _consecutive_failures
    _consecutive_failures = 0


def record_failure() -> None:
    """Increment backoff counter after a failed API call."""
    global _consecutive_failures
    _consecutive_failures += 1
    get_logger().debug("Rate limiter backoff level: %s", _consecutive_failures)
