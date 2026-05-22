"""Phase 1 stub Image Agent."""

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.image import ImageAsset
from auto_book.utils.logger import get_logger


def run_image_agent(
    chapters: list[ChapterDraft],
    book_bible: BookBible,
    output_dir: str,
) -> list[ImageAsset]:
    """Return no images in Phase 1."""

    logger = get_logger()
    logger.info(
        "Image Agent is stubbed in Phase 1 (%s chapters, output: %s)",
        len(chapters),
        output_dir,
    )
    return []
