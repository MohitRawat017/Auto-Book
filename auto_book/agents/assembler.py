"""Phase 1 stub Assembler."""

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.export import ExportResult
from auto_book.utils.logger import get_logger


def assemble_stub(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
) -> ExportResult:
    """Return an export result without writing final book files in Phase 1."""

    logger = get_logger()
    total_words = sum(chapter.word_count for chapter in chapters)
    logger.info(
        "Stub assembler saw %s accepted chapters for '%s'",
        len(chapters),
        book_bible.working_title,
    )
    return ExportResult(
        total_chapters=len(chapters),
        total_word_count=total_words,
        success=True,
    )
