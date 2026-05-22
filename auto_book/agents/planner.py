"""Phase 1 stub Planner Agent."""

from auto_book.config import settings
from auto_book.models.book_bible import BookBible, ChapterOutline
from auto_book.utils.logger import get_logger


def run_planner(user_brief: str, genre: str) -> BookBible:
    """Create a deterministic stub Book Bible."""

    logger = get_logger()
    topic = user_brief.strip() or "Untitled Book"
    chapter_count = max(1, settings.book.default_chapter_count)

    outlines = []
    for index in range(1, chapter_count + 1):
        outlines.append(
            ChapterOutline(
                chapter_number=index,
                title=f"Stub Chapter {index}",
                summary=(
                    f"Phase 1 placeholder chapter {index} for the book idea: "
                    f"{topic}."
                ),
                key_topics=[f"Topic {index}", "Phase 1 skeleton"],
                word_count_target=settings.book.target_words_per_chapter,
                depends_on=[index - 1] if index > 1 else [],
            )
        )

    bible = BookBible(
        working_title=f"Stub Book: {topic[:60]}",
        subtitle="A Phase 1 skeleton plan",
        genre=genre or "non-fiction how-to",
        core_thesis=(
            "This placeholder thesis proves that planning state can flow "
            "through the orchestrator before live LLM generation is added."
        ),
        target_audience="Phase 1 developers and testers",
        reader_pain_points=[
            "Need a runnable skeleton",
            "Need checkpointable state",
            "Need valid model objects for later phases",
        ],
        book_promise="A structurally valid book plan for testing the pipeline.",
        tone="Clear, practical, and implementation-focused",
        style_guide="Use simple Markdown and deterministic stub content.",
        chapter_outline=outlines,
        required_themes=["orchestration", "checkpointing", "valid state"],
        forbidden_topics=["live API calls in Phase 1"],
        glossary={"Book Bible": "Stable plan used to guide a book run."},
        image_direction="Images are deferred beyond Phase 1.",
    )
    logger.info("Stub Book Bible created with %s chapters", len(outlines))
    return bible
