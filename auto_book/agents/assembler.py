"""Markdown assembler for accepted chapters."""

from pathlib import Path
import re

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.export import ExportResult
from auto_book.utils.logger import get_logger


def assemble_markdown(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
    output_dir: str,
) -> ExportResult:
    """Assemble accepted chapters into output/book.md."""

    logger = get_logger()
    output_path = Path(output_dir) / "book.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(chapters, key=lambda chapter: chapter.chapter_number)
    parts: list[str] = []

    parts.append(f"# {book_bible.working_title}\n")
    if book_bible.subtitle:
        parts.append(f"### {book_bible.subtitle}\n")
    parts.append(f"*{book_bible.book_promise}*\n")
    parts.append("---\n")

    parts.append("## Table of Contents\n")
    for chapter in ordered:
        parts.append(
            f"- [Chapter {chapter.chapter_number}: {chapter.title}]"
            f"(#{_anchor(chapter.title)})"
        )
    parts.append("\n---\n")

    for chapter in ordered:
        parts.append(f"\n## Chapter {chapter.chapter_number}: {chapter.title}\n")
        parts.append(_strip_duplicate_title(chapter))
        if chapter.key_takeaways:
            parts.append("\n### Key Takeaways\n")
            parts.extend(f"- {takeaway}" for takeaway in chapter.key_takeaways)
        parts.append("\n---\n")

    content = "\n".join(parts).strip() + "\n"
    output_path.write_text(content, encoding="utf-8")
    word_count = len(content.split())
    logger.info(
        "Markdown assembled at %s (%s chapters, %s words)",
        output_path,
        len(ordered),
        word_count,
    )
    return ExportResult(
        markdown_path=str(output_path),
        total_chapters=len(ordered),
        total_word_count=word_count,
        success=True,
    )


def _anchor(title: str) -> str:
    anchor = title.strip().lower()
    anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)
    anchor = re.sub(r"\s+", "-", anchor)
    return anchor


def _strip_duplicate_title(chapter: ChapterDraft) -> str:
    lines = chapter.body.splitlines()
    if lines and lines[0].strip().startswith("#") and chapter.title.lower() in lines[0].lower():
        return "\n".join(lines[1:]).strip()
    return chapter.body.strip()
