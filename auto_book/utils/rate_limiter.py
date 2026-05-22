"""Simple process-local rate limiter for Groq calls."""

import time

from auto_book.config import settings
from auto_book.utils.logger import get_logger

_last_call_time: float = 0.0


def wait_for_rate_limit() -> None:
    """Wait between LLM calls to reduce free-tier rate-limit pressure."""

    global _last_call_time
    logger = get_logger()
    min_delay = max(0, settings.retry.retry_delay_seconds)
    elapsed = time.time() - _last_call_time
    if elapsed < min_delay:
        wait_time = min_delay - elapsed
        logger.debug("Rate limiter sleeping for %.2fs", wait_time)
        time.sleep(wait_time)
    _last_call_time = time.time()
