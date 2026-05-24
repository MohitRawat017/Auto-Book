"""Assemblers for Markdown and DOCX book exports."""

from pathlib import Path
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.export import ExportResult
from auto_book.models.image import ImageAsset
from auto_book.utils.logger import get_logger

IMAGE_ANCHOR_PATTERN = re.compile(r"^\[IMAGE_ANCHOR:\s*([A-Z0-9_]+)\]\s*$")


def assemble_markdown(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
    output_dir: str,
    image_assets: list[ImageAsset] | None = None,
) -> ExportResult:
    """Assemble accepted chapters into output/book.md with image references."""

    logger = get_logger()
    output_path = Path(output_dir) / "book.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(chapters, key=lambda chapter: chapter.chapter_number)
    images = image_assets or []
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
        parts.append(
            _replace_markdown_anchors(
                _strip_duplicate_title(chapter),
                images,
                output_path.parent,
            )
        )
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
        images_embedded=sum(1 for asset in images if not asset.is_placeholder),
        success=True,
    )


def assemble_docx(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
    output_dir: str,
    image_assets: list[ImageAsset] | None = None,
) -> str:
    """Assemble accepted chapters and available images into output/book.docx."""

    logger = get_logger()
    output_path = Path(output_dir) / "book.docx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(chapters, key=lambda chapter: chapter.chapter_number)
    images = image_assets or []
    doc = Document()

    _add_title_page(doc, book_bible)
    doc.add_page_break()
    doc.add_heading("Table of Contents", level=1)
    for chapter in ordered:
        doc.add_paragraph(
            f"Chapter {chapter.chapter_number}: {chapter.title}",
            style="List Number",
        )

    for chapter in ordered:
        doc.add_page_break()
        _add_chapter(doc, chapter, images)

    doc.save(str(output_path))
    logger.info("DOCX assembled at %s", output_path)
    return str(output_path)


def _add_title_page(doc: Document, bible: BookBible) -> None:
    for _ in range(6):
        doc.add_paragraph("")

    title = doc.add_heading(bible.working_title, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if bible.subtitle:
        subtitle = doc.add_paragraph(bible.subtitle)
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in subtitle.runs:
            run.font.size = Pt(16)
            run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph("")
    tagline = doc.add_paragraph(bible.book_promise)
    tagline.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in tagline.runs:
        run.font.italic = True
        run.font.size = Pt(12)


def _add_chapter(
    doc: Document,
    chapter: ChapterDraft,
    image_assets: list[ImageAsset],
) -> None:
    doc.add_heading(f"Chapter {chapter.chapter_number}: {chapter.title}", level=1)
    _add_markdown_body(doc, _strip_duplicate_title(chapter), image_assets)

    if chapter.key_takeaways:
        doc.add_heading("Key Takeaways", level=3)
        for takeaway in chapter.key_takeaways:
            doc.add_paragraph(takeaway, style="List Bullet")


def _add_markdown_body(
    doc: Document,
    markdown: str,
    image_assets: list[ImageAsset],
) -> None:
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        anchor_match = IMAGE_ANCHOR_PATTERN.fullmatch(line)
        if anchor_match:
            _add_docx_anchor_image(doc, anchor_match.group(1), image_assets)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:], style="List Bullet")
        elif re.match(r"^\d+[\.)]\s+", line):
            doc.add_paragraph(re.sub(r"^\d+[\.)]\s+", "", line), style="List Number")
        else:
            paragraph = doc.add_paragraph()
            _add_formatted_text(paragraph, line)


def _add_formatted_text(paragraph, text: str) -> None:
    parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        else:
            paragraph.add_run(part)


def _replace_markdown_anchors(
    markdown: str,
    image_assets: list[ImageAsset],
    markdown_dir: Path,
) -> str:
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        match = IMAGE_ANCHOR_PATTERN.fullmatch(line)
        if not match:
            lines.append(raw_line)
            continue

        lines.append(_markdown_for_anchor(match.group(1), image_assets, markdown_dir))
    return "\n".join(lines).strip()


def _markdown_for_anchor(
    anchor_id: str,
    image_assets: list[ImageAsset],
    markdown_dir: Path,
) -> str:
    image = _image_for_anchor(anchor_id, image_assets)
    if image is None or image.is_placeholder or not image.file_path:
        label = (
            image.alt_text if image else ""
        ) or (
            image.prompt_used if image else ""
        ) or anchor_id.replace("_", " ").title()
        return f"*[Image placeholder: {label}]*"

    image_path = Path(image.file_path)
    try:
        display_path = image_path.relative_to(markdown_dir)
    except ValueError:
        display_path = image_path
    alt = image.alt_text or anchor_id.replace("_", " ").title()
    return f"![{alt}]({display_path.as_posix()})"


def _add_docx_anchor_image(
    doc: Document,
    anchor_id: str,
    image_assets: list[ImageAsset],
) -> None:
    image = _image_for_anchor(anchor_id, image_assets)
    if image is None or image.is_placeholder or not image.file_path:
        label = (
            image.alt_text if image else ""
        ) or (
            image.prompt_used if image else ""
        ) or anchor_id.replace("_", " ").title()
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(f"[Image placeholder: {label}]")
        run.italic = True
        return

    image_path = Path(image.file_path)
    if not image_path.exists():
        paragraph = doc.add_paragraph()
        run = paragraph.add_run(f"[Image missing: {image.alt_text or anchor_id}]")
        run.italic = True
        return

    doc.add_paragraph("")
    doc.add_picture(str(image_path), width=Inches(5.5))
    if image.alt_text:
        caption = doc.add_paragraph(image.alt_text)
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in caption.runs:
            run.font.italic = True
            run.font.size = Pt(9)


def _image_for_anchor(
    anchor_id: str,
    image_assets: list[ImageAsset],
) -> ImageAsset | None:
    for image in image_assets:
        if image.anchor_id == anchor_id:
            return image
    return None


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
