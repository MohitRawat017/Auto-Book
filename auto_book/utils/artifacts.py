"""Helpers for saving human-inspectable run artifacts."""

from pathlib import Path

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.memory import DynamicMemory
from auto_book.models.review import ReviewDecision
from auto_book.utils.logger import get_logger


def save_book_bible(book_bible: BookBible, output_dir: str) -> str:
    """Save Book Bible JSON."""

    path = Path(output_dir) / "book_bible.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(book_bible.model_dump_json(indent=2), encoding="utf-8")
    get_logger().info("Book Bible saved to %s", path)
    return str(path)


def save_chapter_draft(draft: ChapterDraft, output_dir: str) -> str:
    """Save accepted chapter Markdown."""

    path = Path(output_dir) / "chapters" / f"chapter_{draft.chapter_number:02d}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(draft.body, encoding="utf-8")
    get_logger().info("Chapter %s saved to %s", draft.chapter_number, path)
    return str(path)


def save_review(review: ReviewDecision, output_dir: str) -> str:
    """Save chapter review JSON."""

    path = (
        Path(output_dir)
        / "reviews"
        / f"chapter_{review.chapter_number:02d}_review.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    get_logger().info("Review for chapter %s saved to %s", review.chapter_number, path)
    return str(path)


def save_dynamic_memory(memory: DynamicMemory, output_dir: str) -> str:
    """Save Dynamic Memory JSON."""

    path = Path(output_dir) / "dynamic_memory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")
    get_logger().info("Dynamic Memory saved to %s", path)
    return str(path)
