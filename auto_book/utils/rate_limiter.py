"""Simple process-local rate limiter for Groq calls."""

import math
import re
import time

from auto_book.config import settings
from auto_book.utils.logger import get_logger

_last_call_time: float = 0.0
_consecutive_failures: int = 0

_RETRY_AFTER_PATTERN = re.compile(
    r"try again in\s+"
    r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
    re.IGNORECASE,
)


class RateLimitPause(RuntimeError):
    """Raised when a provider asks the run to pause before retrying."""

    def __init__(
        self,
        message: str,
        retry_after_seconds: int = 0,
        provider: str = "groq",
    ) -> None:
        self.retry_after_seconds = max(0, retry_after_seconds)
        self.provider = provider
        super().__init__(message)


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


def extract_retry_after_seconds(message: str) -> int | None:
    """Extract a provider retry delay from messages like 'try again in 42m44s'."""

    match = _RETRY_AFTER_PATTERN.search(message)
    if not match:
        return None

    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        return None
    return int(math.ceil(total))


def is_rate_limit_error(error: object) -> bool:
    """Return whether an exception/message looks like a provider rate limit."""

    if getattr(error, "status_code", None) == 429:
        return True
    message = str(error).lower()
    return (
        "rate_limit_exceeded" in message
        or "rate_limit" in message
        or "error code: 429" in message
        or "tokens per day" in message
        or "tpd" in message
        or ("429" in message and ("rate" in message or "limit" in message))
    )


def rate_limit_pause_from_exception(exc: Exception) -> RateLimitPause | None:
    """Convert provider rate-limit exceptions into a resumable pause error."""

    if not is_rate_limit_error(exc):
        return None

    retry_after = extract_retry_after_seconds(str(exc))
    if retry_after is None:
        retry_after = max(60, settings.retry.retry_delay_seconds)

    retry_after += 5
    return RateLimitPause(
        "Groq rate limit reached. "
        f"Retry after about {retry_after} seconds. Original error: {exc}",
        retry_after_seconds=retry_after,
        provider="groq",
    )


def raise_if_rate_limited(exc: Exception) -> None:
    """Raise a RateLimitPause when an exception is a provider rate limit."""

    pause = rate_limit_pause_from_exception(exc)
    if pause is None:
        return

    record_failure()
    raise pause from exc
