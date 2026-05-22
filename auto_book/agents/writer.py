"""Phase 1 stub Writer Agent."""

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft, ChapterPlan
from auto_book.models.memory import DynamicMemory
from auto_book.utils.logger import get_logger


def run_writer(
    chapter_plan: ChapterPlan,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
    revision_feedback: list[str] | None = None,
) -> ChapterDraft:
    """Create a deterministic stub chapter draft."""

    logger = get_logger()
    previous_count = len(dynamic_memory.chapters)
    revision_note = ""
    if revision_feedback:
        revision_note = "\n\nRevision feedback received:\n" + "\n".join(
            f"- {item}" for item in revision_feedback
        )

    body = (
        f"# Chapter {chapter_plan.chapter_number}: {chapter_plan.title}\n\n"
        f"This is Phase 1 stub content for **{book_bible.working_title}**. "
        f"The chapter goal is: {chapter_plan.summary}\n\n"
        "## What This Stub Proves\n\n"
        "This draft proves that the writer node can receive a chapter plan, "
        "the Book Bible, and Dynamic Memory, then return a real ChapterDraft "
        "object instead of `None`.\n\n"
        "## Continuity\n\n"
        f"The pipeline has already accepted {previous_count} chapter(s). "
        "Later phases will replace this deterministic text with live LLM "
        "generation while preserving the same state shape."
        f"{revision_note}\n"
    )

    draft = ChapterDraft(
        chapter_number=chapter_plan.chapter_number,
        title=chapter_plan.title,
        body=body,
        key_takeaways=[
            "The writer node returns a valid ChapterDraft.",
            "Dynamic Memory is available to the writer.",
            "Live generation is deferred to Phase 2.",
        ],
        word_count=len(body.split()),
    )
    logger.info(
        "Stub draft created for chapter %s (%s words)",
        draft.chapter_number,
        draft.word_count,
    )
    return draft
