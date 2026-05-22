# Phase 3 — Export & Image Agent

## Goal

Add DOCX export capability and a working Image Agent that generates image prompts and calls the kie.ai API to produce actual images. Images should be embedded in both Markdown (as links) and DOCX (as embedded files).

**Prerequisite**: Phase 2 is complete — the system can produce a full Markdown book autonomously.

**Success Criteria**: The system produces both `book.md` and `book.docx` with embedded images. If image generation fails for any image, the book still exports cleanly with text placeholders.

---

## 3.1 — New Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing Phase 1/2 deps ...
    "python-docx>=1.1.0",       # DOCX generation
    "httpx>=0.27.0",            # HTTP client for kie.ai API calls
    "Pillow>=10.0",             # Image processing (resize, format convert)
]
```

Run `uv sync` after updating.

---

## 3.2 — Image Agent

### File: `auto_book/agents/image_agent.py`

The Image Agent has two responsibilities:
1. **Planning**: Analyze accepted chapters and the Book Bible to decide where images would help, then generate image prompts.
2. **Generation**: Call the kie.ai API with those prompts to produce actual images.

#### 3.2.1 — Image Planning (LLM-powered)

```python
"""
Image Agent — plans image placements and generates images via kie.ai.

Phase 1: Analyze chapters and produce ImagePrompt objects.
Phase 2: Call kie.ai API to generate actual images.
"""

import time
import httpx
from pathlib import Path
from auto_book.agents.llm_client import get_llm
from auto_book.models.image import ImagePrompt, ImageAsset
from auto_book.models.chapter import ChapterDraft
from auto_book.models.book_bible import BookBible
from auto_book.config import secrets, settings
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import wait_for_rate_limit


IMAGE_PLANNER_SYSTEM_PROMPT = """You are an art director for a book project.

Your job is to identify 1-3 places in a chapter where an image would enhance 
the reader's understanding or engagement. Then create specific image generation prompts.

BOOK CONTEXT:
Title: {title}
Genre: {genre}
Image Direction: {image_direction}

RULES:
- Only suggest images that genuinely add value — not decorative filler.
- Each prompt should be detailed enough for an AI image generator.
- Include style guidance (e.g., "minimalist diagram", "watercolor illustration").
- Provide alt text for accessibility.
- For non-fiction: prefer diagrams, charts, infographics, concept illustrations.
- For fiction: prefer scene illustrations, character moments, atmospheric images.
- Position should describe where in the chapter the image belongs 
  (e.g., "after the introduction paragraph", "alongside the comparison table").
"""

IMAGE_PLANNER_USER_PROMPT = """Analyze this chapter and suggest image placements:

Chapter {chapter_number}: "{title}"

{body_excerpt}

Return a list of image prompts (1 to 3 images maximum).
"""


# Pydantic model for LLM structured output
from pydantic import BaseModel, Field

class ImagePlanResponse(BaseModel):
    """Structured output from the image planning LLM call."""
    images: list[ImagePrompt] = Field(default_factory=list)


def plan_images_for_chapter(
    draft: ChapterDraft,
    book_bible: BookBible,
) -> list[ImagePrompt]:
    """
    Analyze a chapter and generate image prompts.

    Args:
        draft: The accepted chapter draft.
        book_bible: For style/genre context.

    Returns:
        A list of ImagePrompt objects (0-3 per chapter).
    """
    logger = get_logger()
    wait_for_rate_limit()

    llm = get_llm("reviewer")  # Use cheaper model for image planning
    structured_llm = llm.with_structured_output(ImagePlanResponse)

    system_msg = IMAGE_PLANNER_SYSTEM_PROMPT.format(
        title=book_bible.working_title,
        genre=book_bible.genre,
        image_direction=book_bible.image_direction or "Use clean, professional illustrations",
    )
    user_msg = IMAGE_PLANNER_USER_PROMPT.format(
        chapter_number=draft.chapter_number,
        title=draft.title,
        body_excerpt=draft.body[:3000],  # Cap excerpt to save tokens
    )

    try:
        result = structured_llm.invoke([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])

        # Ensure chapter numbers are set
        for img in result.images:
            img.chapter_number = draft.chapter_number

        logger.info(f"Image plan for chapter {draft.chapter_number}: {len(result.images)} images")
        return result.images

    except Exception as e:
        logger.warning(f"Image planning failed for chapter {draft.chapter_number}: {e}")
        return []  # Graceful degradation — no images is fine
```

#### 3.2.2 — kie.ai Image Generation

```python
def generate_image_via_kie(
    prompt: ImagePrompt,
    output_dir: str,
    model: str = "flux",  # Or another model available on kie.ai
) -> ImageAsset:
    """
    Call kie.ai API to generate an image from a prompt.

    Args:
        prompt: The ImagePrompt with generation instructions.
        output_dir: Directory to save the generated image.
        model: The kie.ai model to use.

    Returns:
        An ImageAsset with the file path (or placeholder if generation failed).
    """
    logger = get_logger()
    api_key = secrets.kie_api_key

    if not api_key:
        logger.warning("KIE_API_KEY not set — returning placeholder")
        return ImageAsset(
            chapter_number=prompt.chapter_number,
            position=prompt.position,
            prompt_used=prompt.prompt,
            alt_text=prompt.alt_text,
            is_placeholder=True,
        )

    # Prepare output path
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ch{prompt.chapter_number:02d}_{prompt.position[:20].replace(' ', '_')}.png"
    file_path = images_dir / filename

    try:
        # kie.ai API call
        # NOTE: The implementer must check kie.ai docs (https://docs.kie.ai) 
        # for the exact endpoint, request format, and response format.
        # The structure below is a reasonable template based on typical image gen APIs.
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.kie.ai/v1/images/generations",  # Verify exact endpoint
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "prompt": prompt.prompt,
                    "n": 1,
                    "size": "1024x1024",
                },
            )
            response.raise_for_status()
            result = response.json()

            # Download the image
            # NOTE: Response format may vary. Check kie.ai docs for actual structure.
            image_url = result.get("data", [{}])[0].get("url", "")
            if image_url:
                img_response = client.get(image_url)
                img_response.raise_for_status()
                file_path.write_bytes(img_response.content)

                logger.info(f"Image generated: {file_path}")
                return ImageAsset(
                    chapter_number=prompt.chapter_number,
                    position=prompt.position,
                    file_path=str(file_path),
                    prompt_used=prompt.prompt,
                    alt_text=prompt.alt_text,
                    is_placeholder=False,
                )
            else:
                raise ValueError("No image URL in kie.ai response")

    except Exception as e:
        logger.warning(f"Image generation failed: {e}. Using placeholder.")
        return ImageAsset(
            chapter_number=prompt.chapter_number,
            position=prompt.position,
            prompt_used=prompt.prompt,
            alt_text=prompt.alt_text,
            is_placeholder=True,
        )


def run_image_agent(
    chapters: list[ChapterDraft],
    book_bible: BookBible,
    output_dir: str,
) -> list[ImageAsset]:
    """
    Run the full image pipeline: plan images for all chapters, then generate them.

    Args:
        chapters: All accepted chapter drafts.
        book_bible: For context.
        output_dir: Where to save images.

    Returns:
        List of ImageAsset objects (some may be placeholders).
    """
    logger = get_logger()
    all_assets = []

    # Step 1: Plan images for each chapter
    all_prompts: list[ImagePrompt] = []
    for ch in chapters:
        prompts = plan_images_for_chapter(ch, book_bible)
        all_prompts.extend(prompts)

    logger.info(f"Total image prompts generated: {len(all_prompts)}")

    # Save image plan
    plan_path = Path(output_dir) / "images" / "image_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    plan_data = [p.model_dump() for p in all_prompts]
    plan_path.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

    # Step 2: Generate images
    for prompt in all_prompts:
        asset = generate_image_via_kie(prompt, output_dir)
        all_assets.append(asset)
        time.sleep(2)  # Rate limit between image generation calls

    logger.info(
        f"Image generation complete: {sum(1 for a in all_assets if not a.is_placeholder)} "
        f"generated, {sum(1 for a in all_assets if a.is_placeholder)} placeholders"
    )
    return all_assets
```

#### 3.2.3 — kie.ai Integration Notes

> **IMPORTANT FOR IMPLEMENTER**: Before implementing the kie.ai API call:
> 1. Read the official docs at https://docs.kie.ai
> 2. Verify the exact API endpoint URL
> 3. Verify the request body format (model names, parameters)
> 4. Verify the response format (how image URLs are returned)
> 5. Check rate limits and pricing for the chosen model
> 6. Test with the kie.ai playground first
>
> The `generate_image_via_kie()` function above uses a **template structure** based 
> on standard OpenAI-compatible image APIs. The actual kie.ai API may differ.
> Update the endpoint URL, request body, and response parsing as needed.

---

## 3.3 — DOCX Assembler

### File: `auto_book/agents/assembler.py` (UPDATE existing file)

Add a `assemble_docx()` function alongside the existing `assemble_markdown()`.

```python
"""
DOCX Assembly — convert the book into a Word document.

Uses python-docx to build a DOCX file from chapter data.
This is NOT a Markdown-to-DOCX converter. It builds the DOCX structure directly
from the Pydantic models to maintain full control over formatting.
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.image import ImageAsset
from auto_book.utils.logger import get_logger


def assemble_docx(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
    image_assets: list[ImageAsset],
    output_dir: str,
) -> str:
    """
    Assemble all chapters into a DOCX file.

    Args:
        book_bible: The Book Bible for title/metadata.
        chapters: List of accepted chapter drafts in order.
        image_assets: List of generated/placeholder image assets.
        output_dir: Directory to write the output file.

    Returns:
        The file path of the generated DOCX file.
    """
    logger = get_logger()
    output_path = Path(output_dir) / "book.docx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # ── Title Page ──
    _add_title_page(doc, book_bible)

    # ── Table of Contents placeholder ──
    # python-docx doesn't support auto-generated TOC.
    # We add a manual one.
    doc.add_page_break()
    doc.add_heading("Table of Contents", level=1)
    for ch in sorted(chapters, key=lambda c: c.chapter_number):
        p = doc.add_paragraph()
        p.text = f"Chapter {ch.chapter_number}: {ch.title}"
        p.style = "List Number"

    # ── Chapters ──
    for ch in sorted(chapters, key=lambda c: c.chapter_number):
        doc.add_page_break()
        _add_chapter(doc, ch, image_assets)

    # Save
    doc.save(str(output_path))
    logger.info(f"DOCX assembled: {output_path}")
    return str(output_path)


def _add_title_page(doc: Document, bible: BookBible):
    """Add a title page to the DOCX."""
    # Add some spacing
    for _ in range(6):
        doc.add_paragraph("")

    # Title
    title = doc.add_heading(bible.working_title, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Subtitle
    if bible.subtitle:
        subtitle = doc.add_paragraph(bible.subtitle)
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in subtitle.runs:
            run.font.size = Pt(16)
            run.font.color.rgb = RGBColor(100, 100, 100)

    # Spacing
    doc.add_paragraph("")
    doc.add_paragraph("")

    # Book promise as tagline
    tagline = doc.add_paragraph(bible.book_promise)
    tagline.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in tagline.runs:
        run.font.italic = True
        run.font.size = Pt(12)


def _add_chapter(
    doc: Document,
    chapter: ChapterDraft,
    image_assets: list[ImageAsset],
):
    """Add a single chapter to the DOCX."""
    # Chapter heading
    doc.add_heading(f"Chapter {chapter.chapter_number}: {chapter.title}", level=1)

    # Parse the chapter body (simple Markdown-to-DOCX conversion)
    # Handle: paragraphs, ## headings, bullet points, bold, italic
    lines = chapter.body.split("\n")
    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Heading level 2 (##)
        if stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)

        # Heading level 3 (###)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)

        # Bullet point
        elif stripped.startswith("- ") or stripped.startswith("* "):
            doc.add_paragraph(stripped[2:], style="List Bullet")

        # Numbered list
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)" :
            doc.add_paragraph(stripped[2:].strip(), style="List Number")

        # Skip the chapter title if it's repeated as a markdown heading
        elif stripped.startswith("# ") and chapter.title in stripped:
            continue

        # Regular paragraph
        else:
            p = doc.add_paragraph()
            # Handle basic inline markdown: **bold** and *italic*
            _add_formatted_text(p, stripped)

    # Insert chapter images
    chapter_images = [
        img for img in image_assets
        if img.chapter_number == chapter.chapter_number and not img.is_placeholder
    ]
    for img in chapter_images:
        img_path = Path(img.file_path)
        if img_path.exists():
            doc.add_paragraph("")  # Spacing
            doc.add_picture(str(img_path), width=Inches(5.0))
            if img.alt_text:
                caption = doc.add_paragraph(img.alt_text)
                caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in caption.runs:
                    run.font.italic = True
                    run.font.size = Pt(9)

    # Key takeaways
    if chapter.key_takeaways:
        doc.add_heading("Key Takeaways", level=3)
        for takeaway in chapter.key_takeaways:
            doc.add_paragraph(takeaway, style="List Bullet")


def _add_formatted_text(paragraph, text: str):
    """
    Parse basic inline Markdown (**bold**, *italic*) and add formatted runs.
    This is a simple parser — not a full Markdown engine.
    """
    import re

    # Pattern to match **bold** and *italic*
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        else:
            paragraph.add_run(part)
```

---

## 3.4 — Updated Markdown Assembler with Images

Update `assemble_markdown()` in `auto_book/agents/assembler.py` to embed image references:

```python
def assemble_markdown(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
    image_assets: list[ImageAsset],  # NEW parameter
    output_dir: str,
) -> str:
    """Assemble Markdown book with image references."""
    # ... existing title/TOC logic ...

    # When writing each chapter, insert image references
    for ch in sorted(chapters, key=lambda c: c.chapter_number):
        parts.append(f"\n## Chapter {ch.chapter_number}: {ch.title}\n")
        parts.append(ch.body)

        # Insert images for this chapter
        chapter_images = [
            img for img in image_assets
            if img.chapter_number == ch.chapter_number
        ]
        for img in chapter_images:
            if not img.is_placeholder and img.file_path:
                parts.append(f"\n![{img.alt_text}]({img.file_path})\n")
            else:
                parts.append(f"\n*[Image placeholder: {img.alt_text}]*\n")

        # Key takeaways
        if ch.key_takeaways:
            parts.append("\n### Key Takeaways\n")
            for takeaway in ch.key_takeaways:
                parts.append(f"- {takeaway}")

        parts.append("\n---\n")

    # ... rest of assembly ...
```

---

## 3.5 — Orchestrator Updates

### Update `orchestrator/graph.py`

Update the `assemble_book` and `export_book` nodes:

```python
def assemble_book(state: GraphState) -> dict:
    """Node 8: Run Image Agent + Assembler."""
    logger = get_logger()

    # Collect all accepted chapters
    accepted = []
    for cs in state.get("chapter_statuses", []):
        if cs.accepted_draft:
            accepted.append(cs.accepted_draft)
    
    if not accepted:
        return {"error": "No accepted chapters to assemble", "phase": RunPhase.FAILED}

    bible = state["book_bible"]
    output_dir = state.get("output_directory", "./output")  # get from RunState

    # Run Image Agent
    from auto_book.agents.image_agent import run_image_agent
    try:
        image_assets = run_image_agent(accepted, bible, output_dir)
    except Exception as e:
        logger.warning(f"Image agent failed, proceeding without images: {e}")
        image_assets = []

    return {
        "image_assets": image_assets,
        "phase": RunPhase.ASSEMBLING,
    }


def export_book(state: GraphState) -> dict:
    """Node 9: Export to Markdown and DOCX."""
    logger = get_logger()
    from auto_book.agents.assembler import assemble_markdown, assemble_docx

    accepted = []
    for cs in state.get("chapter_statuses", []):
        if cs.accepted_draft:
            accepted.append(cs.accepted_draft)

    bible = state["book_bible"]
    image_assets = state.get("image_assets", [])
    output_dir = state.get("output_directory", "./output")

    errors = []

    # Markdown
    try:
        md_path = assemble_markdown(bible, accepted, image_assets, output_dir)
    except Exception as e:
        md_path = ""
        errors.append(f"Markdown export failed: {e}")

    # DOCX
    try:
        docx_path = assemble_docx(bible, accepted, image_assets, output_dir)
    except Exception as e:
        docx_path = ""
        errors.append(f"DOCX export failed: {e}")

    export_result = ExportResult(
        markdown_path=md_path,
        docx_path=docx_path,
        total_chapters=len(accepted),
        total_word_count=sum(ch.word_count for ch in accepted),
        images_embedded=sum(1 for img in image_assets if not img.is_placeholder),
        success=len(errors) == 0,
        errors=errors,
    )

    return {
        "export_result": export_result,
        "phase": RunPhase.COMPLETED,
    }
```

---

## 3.6 — Config Updates

Add image-related settings to `config.yaml`:

```yaml
# Image Settings
images:
  enabled: true
  max_images_per_chapter: 3
  default_model: "flux"          # kie.ai model to use
  default_size: "1024x1024"
  timeout_seconds: 120
  fallback_to_placeholder: true  # If true, never fail the pipeline due to images
```

Add a corresponding Pydantic model in `config.py`:

```python
class ImageSettings(BaseModel):
    enabled: bool = True
    max_images_per_chapter: int = 3
    default_model: str = "flux"
    default_size: str = "1024x1024"
    timeout_seconds: int = 120
    fallback_to_placeholder: bool = True
```

---

## 3.7 — Verification Checklist

After completing Phase 3, verify:

- [ ] Running the full pipeline produces both `output/book.md` and `output/book.docx`.
- [ ] The DOCX file opens correctly in Microsoft Word / LibreOffice Writer.
- [ ] The DOCX has a formatted title page with the book title and subtitle.
- [ ] The DOCX has a table of contents listing all chapters.
- [ ] Chapter headings, subheadings, bullet points, and bold/italic text render correctly in DOCX.
- [ ] `output/images/image_plan.json` contains image prompts for each chapter.
- [ ] If `KIE_API_KEY` is set, actual images are generated and saved to `output/images/`.
- [ ] Generated images appear in both the Markdown (as `![alt](path)`) and DOCX (embedded).
- [ ] If `KIE_API_KEY` is NOT set, the pipeline still completes with text placeholders.
- [ ] If kie.ai returns an error for one image, other images still generate, and the book exports.
- [ ] The DOCX file size is reasonable (not multi-GB due to uncompressed images).
- [ ] Images in DOCX are properly sized (not overflowing page margins).

### Manual Quality Checks

- Open `book.docx` in Word and verify all chapters are present and formatted.
- Verify images are placed in contextually appropriate locations.
- Check that image prompts in `image_plan.json` are relevant to chapter content.
