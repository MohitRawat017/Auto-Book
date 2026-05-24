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

SCORING CALIBRATION — follow these examples precisely:
- Score 9-10 (pass): Excellent. Reads like a polished book chapter. Rich prose,
  strong examples, good flow, meets word target. Very rare.
- Score 7-8 (pass): Good. Covers the topic well, has decent prose, roughly meets
  the word target. Minor suggestions only. THIS IS THE EXPECTED RANGE for
  a competent draft.
- Score 5-6 (revise): Mediocre. Has structural problems like missing key topics,
  significantly under the word target, or reads like a bullet-point outline
  instead of prose. Requires specific fixes.
- Score 0-4 (fail): Terrible. Off-topic, incoherent, or so short it's unusable.
  Needs full regeneration.

DECISION RULES:
- pass (score 7-10): Chapter is ready. Put minor polish suggestions in
  suggested_edits and return pass. DO NOT return revise for subjective
  improvements like "could use more examples" if the chapter already covers
  the topic adequately.
- revise (score 5-6): Chapter has specific, fixable structural problems.
  You MUST list concrete required_fixes. Only request revisions for
  OBJECTIVE issues, not stylistic preferences.
- fail (score 0-4): Chapter misses the goal entirely. Needs full regeneration.

{revision_context}

Book context:
- Title: {title}
- Thesis: {thesis}
- Audience: {audience}
- Tone: {tone}
- Style guide: {style_guide}

Chapter goal:
Chapter {chapter_number}: "{chapter_title}"
Expected content: {chapter_summary}
Word target: {word_target}
"""

REVIEWER_USER_PROMPT = """Review this chapter draft:

{chapter_body}

{continuity_context}

Return a structured ReviewDecision. Be fair — if the chapter covers its topic
with reasonable prose and approximately meets the word target, score it 7+.
"""


def run_reviewer(
    draft: ChapterDraft,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
    revision_count: int = 0,
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
    word_target = outline.word_count_target if outline else settings.book.target_words_per_chapter

    # Revision-aware context: relax standards after first revision
    revision_context = ""
    if revision_count >= 2:
        revision_context = (
            "IMPORTANT: This chapter has already been revised "
            f"{revision_count} time(s). Be MORE LENIENT in your scoring. "
            "If the chapter covers the core topic and is reasonably well-written, "
            "score it 7+ and pass it. Do NOT ask for further revisions unless "
            "there are critical structural problems. Minor improvements should "
            "go in suggested_edits with a pass decision."
        )
    elif revision_count == 1:
        revision_context = (
            "NOTE: This is a revised draft (revision #1). The writer addressed "
            "previous feedback. If the main issues were fixed, score 7+ and pass. "
            "Only request another revision for remaining serious problems."
        )

    system_msg = REVIEWER_SYSTEM_PROMPT.format(
        title=book_bible.working_title,
        thesis=book_bible.core_thesis,
        audience=book_bible.target_audience,
        tone=book_bible.tone,
        style_guide=book_bible.style_guide,
        chapter_number=draft.chapter_number,
        chapter_title=chapter_title,
        chapter_summary=chapter_summary,
        word_target=word_target,
        revision_context=revision_context,
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
