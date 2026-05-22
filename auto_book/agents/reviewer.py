"""Reviewer Agent: evaluates chapter quality."""

import time

from auto_book.agents.llm_client import get_llm
from auto_book.config import settings
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.memory import DynamicMemory
from auto_book.models.review import ReviewDecision
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import wait_for_rate_limit
from auto_book.utils.tokens import count_tokens, truncate_to_budget
from auto_book.utils.validation import validate_review_decision

REVIEWER_SYSTEM_PROMPT = """You are a strict but constructive book editor.

Evaluate a chapter draft against the Book Bible and previous chapter summaries.

Decision rules:
- pass: score 8-10, chapter is ready.
- revise: score 5-7, chapter has specific fixable issues.
- fail: score 0-4, chapter misses the goal and needs full regeneration.

Book context:
- Title: {title}
- Thesis: {thesis}
- Audience: {audience}
- Tone: {tone}
- Style guide: {style_guide}

Chapter goal:
Chapter {chapter_number}: "{chapter_title}"
Expected content: {chapter_summary}
"""

REVIEWER_USER_PROMPT = """Review this chapter draft:

{chapter_body}

{continuity_context}

Return a structured ReviewDecision with specific problems and required fixes
when the decision is revise or fail.
"""


def run_reviewer(
    draft: ChapterDraft,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
) -> ReviewDecision:
    """Run the live Reviewer Agent."""

    logger = get_logger()
    outline = next(
        (
            chapter
            for chapter in book_bible.chapter_outline
            if chapter.chapter_number == draft.chapter_number
        ),
        None,
    )
    chapter_title = outline.title if outline else draft.title
    chapter_summary = outline.summary if outline else "No chapter summary available."

    system_msg = REVIEWER_SYSTEM_PROMPT.format(
        title=book_bible.working_title,
        thesis=book_bible.core_thesis,
        audience=book_bible.target_audience,
        tone=book_bible.tone,
        style_guide=book_bible.style_guide,
        chapter_number=draft.chapter_number,
        chapter_title=chapter_title,
        chapter_summary=chapter_summary,
    )
    user_msg = REVIEWER_USER_PROMPT.format(
        chapter_body=truncate_to_budget(draft.body, settings.context_budget.total_max // 2),
        continuity_context=_build_review_continuity(dynamic_memory),
    )
    logger.info(
        "Reviewer prompt estimate for chapter %s: %s tokens",
        draft.chapter_number,
        count_tokens(system_msg + user_msg),
    )

    llm = get_llm("reviewer")
    structured_llm = llm.with_structured_output(ReviewDecision)

    last_error: Exception | None = None
    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info("Reviewer attempt %s for chapter %s", attempt, draft.chapter_number)
            wait_for_rate_limit()
            result = structured_llm.invoke(
                [
                    ("system", system_msg),
                    ("human", user_msg),
                ]
            )
            if not isinstance(result, ReviewDecision):
                result = ReviewDecision.model_validate(result)

            result.chapter_number = draft.chapter_number
            errors = validate_review_decision(result)
            if errors:
                raise ValueError("; ".join(errors))

            logger.info(
                "Chapter %s review: %s (score %.1f)",
                result.chapter_number,
                result.decision.value,
                result.score,
            )
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Reviewer attempt %s for chapter %s failed: %s",
                attempt,
                draft.chapter_number,
                exc,
            )
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)

    raise ValueError(
        f"Failed to review chapter {draft.chapter_number} after "
        f"{settings.retry.max_validation_retries} attempts. "
        f"Original error: {last_error}"
    )


def _build_review_continuity(memory: DynamicMemory) -> str:
    if not memory.chapters:
        return "Previous chapters: none."

    lines = ["Previous accepted chapters:"]
    for chapter in memory.chapters:
        lines.append(f"- Chapter {chapter.chapter_number}, {chapter.title}: {chapter.summary}")

    if memory.repetition_warnings:
        lines.append("\nKnown repetition risks:")
        lines.extend(f"- {warning}" for warning in memory.repetition_warnings)

    return truncate_to_budget("\n".join(lines), settings.context_budget.dynamic_memory)
