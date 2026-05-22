"""Planner Agent: creates the Book Bible from a user brief."""

import time

from auto_book.agents.llm_client import get_llm
from auto_book.config import settings
from auto_book.models.book_bible import BookBible
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import wait_for_rate_limit
from auto_book.utils.tokens import count_tokens

PLANNER_SYSTEM_PROMPT = """You are an expert book planner and publishing strategist.

Create a complete Book Bible that will guide an autonomous book-writing pipeline.

Rules:
- The book should have exactly {chapter_count} chapters.
- Each chapter should target approximately {words_per_chapter} words.
- The total book should be approximately {total_pages} pages.
- Define a specific target reader, not "everyone".
- Create a clear thesis or central promise that ties the whole book together.
- The tone and style must match the genre.
- Every chapter must have a distinct purpose and a logical place in the sequence.
- Chapter numbers must be sequential starting from 1.
- Include useful forbidden topics to keep the writer on track.
- Include a glossary for terms that need consistency.

Genre: {genre}
"""

PLANNER_USER_PROMPT = """Create a Book Bible for this book idea:

{brief}

Return a complete plan with title, subtitle, genre, thesis, target audience,
reader pain points, book promise, tone, style guide, chapter outline, required
themes, forbidden topics, glossary, and image direction.
"""


def run_planner(user_brief: str, genre: str) -> BookBible:
    """Run the live Planner Agent."""

    logger = get_logger()
    system_msg = PLANNER_SYSTEM_PROMPT.format(
        chapter_count=settings.book.default_chapter_count,
        words_per_chapter=settings.book.target_words_per_chapter,
        total_pages=settings.book.target_total_pages,
        genre=genre,
    )
    user_msg = PLANNER_USER_PROMPT.format(brief=user_brief)
    logger.info("Planner prompt estimate: %s tokens", count_tokens(system_msg + user_msg))

    llm = get_llm("planner")
    structured_llm = llm.with_structured_output(BookBible)

    last_error: Exception | None = None
    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info("Planner attempt %s", attempt)
            wait_for_rate_limit()
            result = structured_llm.invoke(
                [
                    ("system", system_msg),
                    ("human", user_msg),
                ]
            )
            if not isinstance(result, BookBible):
                result = BookBible.model_validate(result)

            errors = _validate_book_bible(result)
            if errors:
                raise ValueError("; ".join(errors))

            logger.info(
                "Book Bible created: '%s' with %s chapters",
                result.working_title,
                len(result.chapter_outline),
            )
            return result
        except Exception as exc:
            last_error = exc
            logger.warning("Planner attempt %s failed: %s", attempt, exc)
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)

    raise ValueError(
        "Failed to create a valid Book Bible after "
        f"{settings.retry.max_validation_retries} attempts. "
        f"Original error: {last_error}"
    )


def _validate_book_bible(book_bible: BookBible) -> list[str]:
    """Semantic validation beyond Pydantic schema checks."""

    errors: list[str] = []
    if not book_bible.working_title.strip():
        errors.append("working_title is empty")
    if not book_bible.core_thesis.strip():
        errors.append("core_thesis is empty")
    if not book_bible.target_audience.strip():
        errors.append("target_audience is empty")
    if not book_bible.book_promise.strip():
        errors.append("book_promise is empty")

    expected = settings.book.default_chapter_count
    actual = len(book_bible.chapter_outline)
    if actual != expected:
        errors.append(f"chapter_outline has {actual} chapters; expected {expected}")

    seen_titles: set[str] = set()
    for index, chapter in enumerate(book_bible.chapter_outline, 1):
        if chapter.chapter_number != index:
            errors.append(
                f"chapter {index} has chapter_number={chapter.chapter_number}"
            )
            break
        normalized = chapter.title.strip().lower()
        if not normalized:
            errors.append(f"chapter {index} title is empty")
        if normalized in seen_titles:
            errors.append(f"duplicate chapter title: {chapter.title}")
        seen_titles.add(normalized)

    return errors
