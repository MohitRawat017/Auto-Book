"""Token counting and context budget helpers."""

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _get_encoding():
    """Load tiktoken encoding lazily, falling back if its cache is unavailable."""

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Count tokens in text."""

    if not text:
        return 0

    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))

    # Offline fallback: rough English-token approximation.
    return max(1, int(len(text.split()) * 1.35))


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Keep the most recent text that fits within max_tokens."""

    encoding = _get_encoding()
    if encoding is None:
        words = text.split()
        approx_words = max(1, int(max_tokens / 1.35))
        return " ".join(words[-approx_words:])

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[-max_tokens:])


def fits_budget(text: str, max_tokens: int) -> bool:
    """Return whether text fits the token budget."""

    return count_tokens(text) <= max_tokens
