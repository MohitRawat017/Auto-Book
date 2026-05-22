"""Phase 1 stub Reviewer Agent."""

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.memory import DynamicMemory
from auto_book.models.review import ReviewDecision, ReviewDecisionEnum
from auto_book.utils.logger import get_logger


def run_reviewer(
    draft: ChapterDraft,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
) -> ReviewDecision:
    """Return a deterministic passing review for Phase 1 stubs."""

    logger = get_logger()
    logger.info(
        "Stub reviewer passing chapter %s for %s",
        draft.chapter_number,
        book_bible.working_title,
    )
    return ReviewDecision(
        chapter_number=draft.chapter_number,
        decision=ReviewDecisionEnum.PASS,
        score=8.0,
        suggested_edits=[
            "Replace stub text with live LLM content in Phase 2.",
        ],
        tone_match=True,
    )
