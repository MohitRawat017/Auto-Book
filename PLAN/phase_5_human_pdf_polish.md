# Phase 5 — Human-in-Loop, PDF, & Polish

## Goal

Add optional human approval checkpoints at key stages, PDF export, genre-specific style presets, a test suite with mock agents, and production hardening (better error messages, documentation, example runs).

**Prerequisite**: Phases 1-4 are complete — the system produces research-backed books with images in Markdown + DOCX.

**Success Criteria**: The system can run in two modes — fully autonomous (default) or human-supervised (with approval gates). PDF export works. A full test suite passes. A new user can install, configure, and run the system from the README alone.

---

## 5.1 — Human-in-Loop Checkpoints

### Design Approach

Human-in-loop is implemented **inside the orchestrator** as optional pause points. When enabled, the orchestrator saves state, prints the artifact for human review, and waits for user input before continuing. When disabled, these checkpoints are silently skipped.

**Checkpoint locations** (in pipeline order):

| Checkpoint | When | What the user reviews |
|---|---|---|
| `approve_bible` | After Planner creates Book Bible | Title, thesis, audience, chapter outline |
| `approve_chapter` | After Reviewer passes a chapter | Chapter content, review score |
| `approve_images` | After Image Agent generates prompts | Image prompts before calling kie.ai |
| `approve_export` | Before final export | Full assembled Markdown |

### Config

Add to `config.yaml`:

```yaml
# Human-in-Loop Settings
human_review:
  enabled: false                  # Set to true to enable approval checkpoints
  approve_bible: true             # Pause after Book Bible creation
  approve_chapters: false         # Pause after each chapter (can be slow)
  approve_images: true            # Pause before image generation
  approve_export: false           # Pause before final export
```

Add corresponding Pydantic model in `config.py`:

```python
class HumanReviewSettings(BaseModel):
    enabled: bool = False
    approve_bible: bool = True
    approve_chapters: bool = False
    approve_images: bool = True
    approve_export: bool = False
```

### Implementation: `auto_book/orchestrator/human_review.py` (NEW)

```python
"""
Human-in-loop review system.

When enabled, pauses the pipeline at configured checkpoints and asks the 
user to approve, edit, or reject the current artifact.

The orchestrator saves state before each checkpoint, so if the user 
closes the terminal, they can resume later with --resume.
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from auto_book.config import settings
from auto_book.utils.logger import get_logger

console = Console()


def should_pause(checkpoint_name: str) -> bool:
    """Check if a given checkpoint should pause for human review."""
    hr = settings.human_review
    if not hr.enabled:
        return False
    return getattr(hr, checkpoint_name, False)


def request_approval(
    checkpoint_name: str,
    title: str,
    content: str,
    allow_edit: bool = False,
) -> dict:
    """
    Display content to the user and request approval.

    Args:
        checkpoint_name: Name of the checkpoint (for logging).
        title: Human-readable title shown to the user.
        content: The content to review (Markdown string).
        allow_edit: If True, offer an "edit" option.

    Returns:
        dict with:
          - "approved": bool
          - "feedback": str (user's feedback if they request changes)
          - "action": str ("approve", "edit", "reject")
    """
    logger = get_logger()
    logger.info(f"Human review checkpoint: {checkpoint_name}")

    console.print()
    console.print(Panel(
        Markdown(content[:3000]),  # Cap display length
        title=f"[bold cyan]Review: {title}[/]",
        border_style="cyan",
        expand=True,
    ))

    if len(content) > 3000:
        console.print(f"[dim](Content truncated for display — full version saved to disk)[/]")

    console.print()

    # Ask for decision
    options = ["approve", "request changes"]
    if allow_edit:
        options.append("edit directly")
    options.append("reject and stop")

    action = Prompt.ask(
        "[bold]Your decision[/]",
        choices=options,
        default="approve",
    )

    if action == "approve":
        console.print("[green]✓ Approved[/]\n")
        return {"approved": True, "feedback": "", "action": "approve"}

    elif action == "request changes":
        feedback = Prompt.ask("[bold]What changes do you want?[/]")
        console.print("[yellow]↻ Changes requested — feeding back into pipeline[/]\n")
        return {"approved": False, "feedback": feedback, "action": "edit"}

    elif action == "reject and stop":
        console.print("[red]✗ Rejected — stopping pipeline[/]\n")
        return {"approved": False, "feedback": "", "action": "reject"}

    return {"approved": True, "feedback": "", "action": "approve"}
```

### Orchestrator Integration

Add checkpoint calls in the graph nodes. Example for Book Bible approval:

```python
# In orchestrator/graph.py, after plan_book creates the bible:

def plan_book(state: GraphState) -> dict:
    """Node 2: Planner Agent creates Book Bible, with optional human approval."""
    logger = get_logger()
    from auto_book.agents.planner import run_planner
    from auto_book.orchestrator.human_review import should_pause, request_approval

    bible = run_planner(state["user_brief"], state["genre"])

    # Human checkpoint
    if should_pause("approve_bible"):
        # Format Bible for display
        display = _format_bible_for_review(bible)
        result = request_approval("approve_bible", "Book Bible", display)

        if result["action"] == "reject":
            return {"error": "User rejected Book Bible", "phase": RunPhase.FAILED}

        if result["action"] == "edit" and result["feedback"]:
            # Re-run planner with user feedback incorporated
            enhanced_brief = (
                state["user_brief"] + 
                "\n\nADDITIONAL DIRECTION FROM EDITOR:\n" + 
                result["feedback"]
            )
            bible = run_planner(enhanced_brief, state["genre"])

    # ... rest of plan_book ...


def _format_bible_for_review(bible) -> str:
    """Format BookBible as readable Markdown for human review."""
    lines = [
        f"# {bible.working_title}",
        f"*{bible.subtitle}*" if bible.subtitle else "",
        f"\n**Genre:** {bible.genre}",
        f"\n**Core Thesis:** {bible.core_thesis}",
        f"\n**Target Audience:** {bible.target_audience}",
        f"\n**Book Promise:** {bible.book_promise}",
        f"\n**Tone:** {bible.tone}",
        f"\n## Chapter Outline\n",
    ]
    for ch in bible.chapter_outline:
        lines.append(f"**{ch.chapter_number}. {ch.title}**")
        lines.append(f"   {ch.summary}\n")
    return "\n".join(lines)
```

Similarly add checkpoints after `review_chapter` (for `approve_chapters`) and before image generation (for `approve_images`).

### Resume After Interrupt

The existing checkpointing system (Phase 1) already saves `RunState` after each step. When a user pauses at a checkpoint and closes the terminal, they can resume with:

```bash
python -m auto_book --resume
```

The orchestrator loads the checkpoint, determines which node to restart from based on `RunPhase`, and continues.

**Implementation note**: Ensure that `save_checkpoint()` is called **before** every `request_approval()` call so state is never lost.

---

## 5.2 — PDF Export

> **DECISION POINT FOR USER**
>
> PDF generation on Windows is notoriously difficult. Options:
>
> | Approach | Pros | Cons |
> |---|---|---|
> | **weasyprint** | Good quality, CSS-styled | Requires GTK/Cairo system deps — painful on Windows |
> | **Pandoc + LaTeX** | Professional typesetting | Requires Pandoc + TeX install (2-4 GB) |
> | **fpdf2** | Pure Python, no deps | Manual layout, no Markdown support, basic output |
> | **md-to-pdf** (Node.js) | Easy, good quality | Requires Node.js installed |
> | **DOCX → PDF via LibreOffice CLI** | Leverages existing DOCX | Requires LibreOffice installed |
> | **pdfkit + wkhtmltopdf** | HTML-to-PDF, decent | Requires wkhtmltopdf binary |
>
> **Recommendation**: Use `fpdf2` for a pure-Python zero-dependency solution.
> Quality won't be as high as LaTeX, but it works everywhere without system deps.
> Alternatively, convert DOCX → PDF via LibreOffice CLI if the user has it installed.
>
> **Ask the user which approach they prefer before implementing.**

### Implementation with fpdf2

Add to `pyproject.toml`:

```toml
    "fpdf2>=2.8.0",              # Pure Python PDF generation
```

### File: `auto_book/agents/pdf_exporter.py` (NEW)

```python
"""
PDF Exporter — generates a PDF book from chapter data.

Uses fpdf2 for pure-Python PDF generation (no system dependencies).
Quality is basic but functional. For professional typesetting, 
consider Pandoc + LaTeX in a future upgrade.
"""

from pathlib import Path
from fpdf import FPDF
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.image import ImageAsset
from auto_book.utils.logger import get_logger


class BookPDF(FPDF):
    """Custom PDF class with header/footer for the book."""

    def __init__(self, title: str = ""):
        super().__init__()
        self.book_title = title

    def header(self):
        if self.page_no() > 1:  # Skip header on title page
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, self.book_title, align="C")
            self.ln(5)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_pdf(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
    image_assets: list[ImageAsset],
    output_dir: str,
) -> str:
    """
    Generate a PDF book.

    Args:
        book_bible: For title/metadata.
        chapters: Accepted chapter drafts.
        image_assets: Generated images (optional).
        output_dir: Where to save the PDF.

    Returns:
        File path of the generated PDF.
    """
    logger = get_logger()
    output_path = Path(output_dir) / "book.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = BookPDF(title=book_bible.working_title)
    pdf.set_auto_page_break(auto=True, margin=25)

    # ── Title Page ──
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 28)
    pdf.multi_cell(0, 15, book_bible.working_title, align="C")
    if book_bible.subtitle:
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 16)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 10, book_bible.subtitle, align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8, book_bible.book_promise, align="C")

    # ── Table of Contents ──
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Table of Contents", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 12)
    for ch in sorted(chapters, key=lambda c: c.chapter_number):
        pdf.cell(0, 8, f"  Chapter {ch.chapter_number}: {ch.title}", ln=True)

    # ── Chapters ──
    for ch in sorted(chapters, key=lambda c: c.chapter_number):
        pdf.add_page()
        _add_chapter_to_pdf(pdf, ch, image_assets)

    # Save
    pdf.output(str(output_path))
    logger.info(f"PDF exported: {output_path}")
    return str(output_path)


def _add_chapter_to_pdf(
    pdf: BookPDF,
    chapter: ChapterDraft,
    image_assets: list[ImageAsset],
):
    """Add a single chapter to the PDF."""
    # Chapter heading
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 12, f"Chapter {chapter.chapter_number}: {chapter.title}")
    pdf.ln(5)

    # Chapter body — simple paragraph-based rendering
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)

    lines = chapter.body.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pdf.ln(4)
            continue

        # Section heading
        if stripped.startswith("## "):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, stripped[3:])
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)

        elif stripped.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, stripped[4:])
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)

        # Bullet point
        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.cell(10)  # Indent
            pdf.multi_cell(0, 6, f"• {stripped[2:]}")

        # Skip markdown heading that repeats chapter title
        elif stripped.startswith("# ") and chapter.title in stripped:
            continue

        # Regular paragraph
        else:
            # Strip basic markdown formatting for PDF
            clean = stripped.replace("**", "").replace("*", "")
            pdf.multi_cell(0, 6, clean)

    # Insert images
    chapter_images = [
        img for img in image_assets
        if img.chapter_number == chapter.chapter_number and not img.is_placeholder
    ]
    for img in chapter_images:
        img_path = Path(img.file_path)
        if img_path.exists():
            try:
                pdf.ln(5)
                pdf.image(str(img_path), w=140)  # 140mm width
                if img.alt_text:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(100, 100, 100)
                    pdf.multi_cell(0, 5, img.alt_text, align="C")
                    pdf.set_font("Helvetica", "", 11)
                    pdf.set_text_color(30, 30, 30)
                pdf.ln(5)
            except Exception:
                pass  # Skip image if it can't be embedded

    # Key takeaways
    if chapter.key_takeaways:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Key Takeaways", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for takeaway in chapter.key_takeaways:
            pdf.cell(10)
            pdf.multi_cell(0, 6, f"• {takeaway}")
```

### Orchestrator Integration

Update `export_book` node to include PDF:

```python
def export_book(state: GraphState) -> dict:
    # ... existing Markdown + DOCX export ...
    
    # PDF
    pdf_path = ""
    try:
        from auto_book.agents.pdf_exporter import export_pdf
        pdf_path = export_pdf(bible, accepted, image_assets, output_dir)
    except Exception as e:
        errors.append(f"PDF export failed: {e}")
        logger.warning(f"PDF export failed (non-fatal): {e}")

    export_result = ExportResult(
        markdown_path=md_path,
        docx_path=docx_path,
        pdf_path=pdf_path,  # NEW
        # ... rest ...
    )
```

---

## 5.3 — Genre-Specific Style Presets

### File: `auto_book/agents/genre_presets.py` (NEW)

Provide default tone, style guide, and planner hints based on genre selection.

```python
"""
Genre presets — provide genre-specific defaults for the Planner Agent.

These presets are injected into the Planner's prompt to guide Book Bible creation.
The user can override any of these in the Book Bible after review.
"""

GENRE_PRESETS = {
    "non-fiction how-to": {
        "tone_hint": "Clear, practical, and encouraging. Use direct address ('you'). "
                     "Break complex topics into actionable steps.",
        "style_hint": "Short paragraphs (3-5 sentences). Use bullet points and numbered "
                      "lists for steps. Include real-world examples. Each chapter should "
                      "end with a clear takeaway or action item.",
        "structure_hint": "Open with why the topic matters, build foundational concepts "
                         "first, progress to advanced topics, end with a practical roadmap.",
        "image_hint": "Diagrams, flowcharts, step-by-step illustrations, and infographics.",
    },
    "non-fiction informational": {
        "tone_hint": "Authoritative but accessible. Avoid jargon unless defined. "
                     "Balance depth with readability.",
        "style_hint": "Mix narrative explanations with data and evidence. Use section "
                      "headings liberally. Include statistics and citations where possible.",
        "structure_hint": "Start with context and history, explore the topic from multiple "
                         "angles, present evidence and analysis, conclude with implications.",
        "image_hint": "Data visualizations, timelines, comparison charts, and photographs.",
    },
    "self-help": {
        "tone_hint": "Warm, empathetic, and motivating. Speak to the reader as a supportive "
                     "mentor. Use personal anecdotes and relatable examples.",
        "style_hint": "Conversational paragraphs. Include reflection questions and exercises. "
                      "Use stories to illustrate points. Each chapter should have a clear "
                      "transformation promise.",
        "structure_hint": "Open with a relatable problem, build awareness, provide frameworks "
                         "and tools, include exercises, close with encouragement.",
        "image_hint": "Motivational illustrations, simple diagrams of frameworks, "
                      "and reflection journal templates.",
    },
    "fiction": {
        "tone_hint": "Match the subgenre. Prioritize voice, atmosphere, and character. "
                     "Show don't tell. Use sensory details.",
        "style_hint": "Vary sentence length for rhythm. Use dialogue to reveal character. "
                      "Avoid exposition dumps. Each chapter should end with a hook or "
                      "turning point.",
        "structure_hint": "Follow three-act structure or chosen narrative framework. "
                         "Build tension progressively. Plant and pay off story threads.",
        "image_hint": "Scene illustrations, character portraits, atmospheric art, and maps "
                      "if the story involves world-building.",
    },
    "memoir": {
        "tone_hint": "Intimate and reflective. Balance vulnerability with insight. "
                     "Use present tense for vivid scenes, past tense for reflection.",
        "style_hint": "Mix scene-based storytelling with reflection. Use concrete sensory "
                      "details. Include dialogue where remembered. Each chapter should "
                      "center on a specific period, theme, or turning point.",
        "structure_hint": "Can be chronological or thematic. Open with a compelling scene. "
                         "Build toward the central transformation or lesson.",
        "image_hint": "Minimal — perhaps chapter header illustrations or period-appropriate art.",
    },
}


def get_genre_preset(genre: str) -> dict:
    """
    Get genre-specific hints for the Planner.
    
    Returns a dict with keys: tone_hint, style_hint, structure_hint, image_hint.
    Falls back to generic hints if genre is not recognized.
    """
    # Try exact match first
    if genre in GENRE_PRESETS:
        return GENRE_PRESETS[genre]

    # Try partial match
    genre_lower = genre.lower()
    for key, preset in GENRE_PRESETS.items():
        if key in genre_lower or genre_lower in key:
            return preset

    # Fallback
    return {
        "tone_hint": "Appropriate for the genre and audience.",
        "style_hint": "Clear, well-structured, and engaging.",
        "structure_hint": "Logical progression with clear chapter purposes.",
        "image_hint": "Use images where they enhance understanding.",
    }
```

### Planner Integration

Update the Planner Agent's system prompt to include genre preset hints:

```python
# In planner.py:
from auto_book.agents.genre_presets import get_genre_preset

def run_planner(user_brief: str, genre: str) -> BookBible:
    preset = get_genre_preset(genre)
    
    system_msg = PLANNER_SYSTEM_PROMPT.format(
        chapter_count=settings.book.default_chapter_count,
        words_per_chapter=settings.book.target_words_per_chapter,
        total_pages=settings.book.target_total_pages,
        genre=genre,
    )
    # Append genre hints
    system_msg += f"\n\nGENRE GUIDANCE:"
    system_msg += f"\n- Tone: {preset['tone_hint']}"
    system_msg += f"\n- Style: {preset['style_hint']}"
    system_msg += f"\n- Structure: {preset['structure_hint']}"
    system_msg += f"\n- Images: {preset['image_hint']}"
    
    # ... rest of planner ...
```

---

## 5.4 — Test Suite

### New Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
]
```

### Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py               # Shared fixtures (mock LLM, mock config)
├── test_models/
│   ├── __init__.py
│   ├── test_book_bible.py
│   ├── test_chapter.py
│   ├── test_review.py
│   ├── test_memory.py
│   └── test_run_state.py
├── test_utils/
│   ├── __init__.py
│   ├── test_tokens.py
│   ├── test_validation.py
│   └── test_research_selector.py
├── test_agents/
│   ├── __init__.py
│   ├── test_planner.py
│   ├── test_writer.py
│   ├── test_reviewer.py
│   ├── test_memory_updater.py
│   └── test_assembler.py
├── test_orchestrator/
│   ├── __init__.py
│   ├── test_graph.py         # Graph structure and routing
│   ├── test_checkpointer.py
│   └── test_state_machine.py # End-to-end with mocks
└── test_e2e/
    ├── __init__.py
    └── test_golden_run.py    # Full pipeline with mocked LLM
```

### Key Test Files

#### `tests/conftest.py` — Shared Fixtures

```python
"""
Shared test fixtures.

Provides mock LLM responses so tests run without hitting Groq API.
"""

import pytest
from unittest.mock import MagicMock, patch
from auto_book.models import (
    BookBible, ChapterOutline, ChapterDraft, ReviewDecision,
    ReviewDecisionEnum, DynamicMemory, ChapterMemoryEntry,
)


@pytest.fixture
def sample_bible():
    """A valid BookBible for testing."""
    return BookBible(
        working_title="Test Book",
        subtitle="A Test Subtitle",
        genre="non-fiction how-to",
        core_thesis="Testing is important",
        target_audience="Developers",
        reader_pain_points=["Don't know how to test", "Tests are slow"],
        book_promise="You will learn to write great tests",
        tone="Clear and practical",
        style_guide="Short paragraphs, code examples",
        chapter_outline=[
            ChapterOutline(
                chapter_number=i,
                title=f"Chapter {i} Title",
                summary=f"Summary for chapter {i}",
                key_topics=[f"Topic {i}a", f"Topic {i}b"],
                word_count_target=1500,
            )
            for i in range(1, 4)  # 3 chapters for fast tests
        ],
        required_themes=["testing", "quality"],
        forbidden_topics=["shortcuts", "skipping tests"],
    )


@pytest.fixture
def sample_draft():
    """A valid ChapterDraft for testing."""
    return ChapterDraft(
        chapter_number=1,
        title="Chapter 1 Title",
        body="This is the chapter body. " * 200,  # ~1000 words
        key_takeaways=["Takeaway 1", "Takeaway 2"],
        word_count=1000,
    )


@pytest.fixture
def sample_review_pass():
    """A passing review decision."""
    return ReviewDecision(
        chapter_number=1,
        decision=ReviewDecisionEnum.PASS,
        score=8.5,
        tone_match=True,
    )


@pytest.fixture
def sample_review_revise():
    """A revision-requesting review decision."""
    return ReviewDecision(
        chapter_number=1,
        decision=ReviewDecisionEnum.REVISE,
        score=5.0,
        problems=["Too short", "Missing examples"],
        required_fixes=["Add 300 more words", "Include a code example"],
        tone_match=True,
    )


@pytest.fixture
def sample_memory():
    """A DynamicMemory with one chapter entry."""
    return DynamicMemory(
        chapters=[
            ChapterMemoryEntry(
                chapter_number=1,
                title="Chapter 1",
                summary="This chapter covered the basics of testing.",
                key_claims=["Testing catches bugs early"],
                definitions_introduced=["unit test", "integration test"],
            )
        ],
        total_word_count=1000,
    )


@pytest.fixture
def mock_config(tmp_path):
    """Patch settings to use a temp directory for output."""
    with patch("auto_book.config.settings") as mock_settings:
        mock_settings.output.directory = str(tmp_path)
        mock_settings.book.min_words_per_chapter = 100
        mock_settings.book.max_words_per_chapter = 5000
        mock_settings.book.default_chapter_count = 3
        mock_settings.book.target_words_per_chapter = 1500
        mock_settings.retry.max_validation_retries = 2
        mock_settings.retry.retry_delay_seconds = 0  # No delays in tests
        mock_settings.context_budget.dynamic_memory = 1500
        mock_settings.context_budget.research_notes = 1000
        mock_settings.context_budget.total_max = 8000
        yield mock_settings
```

#### `tests/test_models/test_book_bible.py` — Model Validation Tests

```python
"""Test BookBible model validation."""

import pytest
from pydantic import ValidationError
from auto_book.models.book_bible import BookBible, ChapterOutline


def test_valid_bible(sample_bible):
    """A valid BookBible should pass validation."""
    assert sample_bible.working_title == "Test Book"
    assert len(sample_bible.chapter_outline) == 3


def test_empty_title_accepted():
    """BookBible with empty title should still be valid at schema level.
    Semantic validation catches this separately."""
    bible = BookBible(
        working_title="",
        genre="test",
        core_thesis="test",
        target_audience="test",
        book_promise="test",
        tone="test",
        style_guide="test",
        chapter_outline=[],
    )
    assert bible.working_title == ""


def test_chapter_outline_ordering(sample_bible):
    """Chapter numbers should be sequential."""
    for i, ch in enumerate(sample_bible.chapter_outline):
        assert ch.chapter_number == i + 1
```

#### `tests/test_orchestrator/test_state_machine.py` — Integration Test

```python
"""
End-to-end orchestrator test with mocked LLM responses.

This test verifies the complete graph flow without hitting any API.
"""

import pytest
from unittest.mock import patch, MagicMock
from auto_book.orchestrator.graph import build_graph
from auto_book.models import (
    BookBible, ChapterOutline, ChapterDraft,
    ReviewDecision, ReviewDecisionEnum,
    DynamicMemory, ChapterMemoryEntry,
)


@pytest.fixture
def mock_all_agents(sample_bible, sample_draft, sample_review_pass, sample_memory):
    """Mock all agent functions to return canned responses."""
    with patch("auto_book.agents.planner.run_planner") as mock_planner, \
         patch("auto_book.agents.writer.run_writer") as mock_writer, \
         patch("auto_book.agents.reviewer.run_reviewer") as mock_reviewer, \
         patch("auto_book.agents.memory_updater.run_memory_update") as mock_memory:

        mock_planner.return_value = sample_bible
        mock_writer.return_value = sample_draft
        mock_reviewer.return_value = sample_review_pass
        mock_memory.return_value = sample_memory

        yield {
            "planner": mock_planner,
            "writer": mock_writer,
            "reviewer": mock_reviewer,
            "memory": mock_memory,
        }


def test_full_pipeline_with_mocks(mock_all_agents):
    """The full pipeline should complete with all mocked agents."""
    graph = build_graph()

    result = graph.invoke({
        "user_brief": "Test book about testing",
        "genre": "non-fiction how-to",
        "dynamic_memory": DynamicMemory(),
        "current_chapter": 0,
        "chapter_statuses": [],
        "phase": "initialized",
        "token_usage": {"total_input_tokens": 0, "total_output_tokens": 0, "calls_made": 0},
        "revision_count": 0,
        "regeneration_count": 0,
        "image_assets": [],
        "error": "",
    })

    assert result["phase"] in ("completed", "COMPLETED")
    assert mock_all_agents["planner"].called
    assert mock_all_agents["writer"].called
    assert mock_all_agents["reviewer"].called


def test_review_retry_loop(mock_all_agents, sample_review_revise, sample_review_pass):
    """A revise decision should trigger a retry, then pass on second attempt."""
    # First call: revise, second call: pass
    mock_all_agents["reviewer"].side_effect = [sample_review_revise, sample_review_pass]
    
    # ... invoke graph and verify writer was called twice for chapter 1 ...
```

#### `tests/test_utils/test_tokens.py` — Token Counting Tests

```python
"""Test token counting utilities."""

from auto_book.utils.tokens import count_tokens, truncate_to_budget, fits_budget


def test_count_tokens_basic():
    """Token count should be positive for non-empty text."""
    assert count_tokens("Hello world") > 0


def test_count_tokens_empty():
    """Empty string should have 0 tokens."""
    assert count_tokens("") == 0


def test_truncate_to_budget():
    """Truncated text should fit within budget."""
    long_text = "word " * 1000
    truncated = truncate_to_budget(long_text, 100)
    assert count_tokens(truncated) <= 100


def test_fits_budget():
    """fits_budget should correctly identify texts within/over budget."""
    assert fits_budget("short", 100)
    assert not fits_budget("word " * 1000, 10)
```

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=auto_book --cov-report=html

# Run a specific test file
pytest tests/test_models/test_book_bible.py -v
```

### Golden Run Test

Save the output of a successful full run (with real LLM calls) as a reference. Store in `tests/fixtures/golden_run/`. Future tests can compare structure (not exact content) against this baseline.

```python
# tests/test_e2e/test_golden_run.py

def test_golden_run_structure():
    """Verify that a saved golden run has the expected file structure."""
    golden_dir = Path("tests/fixtures/golden_run/output")
    assert (golden_dir / "book.md").exists()
    assert (golden_dir / "book_bible.json").exists()
    assert (golden_dir / "dynamic_memory.json").exists()
    assert (golden_dir / "chapters").is_dir()
    assert len(list((golden_dir / "chapters").glob("*.md"))) >= 3
```

---

## 5.5 — Production Polish

### 5.5.1 — Rich CLI Output

Enhance the CLI with progress bars and status panels using `rich`:

```python
# In __main__.py, add a live progress display:

from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Show progress during generation:
# [✓] Planning      Book Bible created: "Personal Finance for Beginners"
# [▶] Writing       Chapter 3/10: "Budgeting Basics" (attempt 1)
# [ ] Reviewing
# [ ] Assembling
# [ ] Exporting
```

### 5.5.2 — README.md

Write a comprehensive README covering:

1. **What it does** — one-paragraph description.
2. **Quick Start** — install, configure, run in 3 steps.
3. **Configuration** — explain `.env` and `config.yaml`.
4. **Usage** — CLI flags and examples.
5. **Architecture** — link to the architecture docs, brief explanation.
6. **Output** — what files are produced and where.
7. **Human Review Mode** — how to enable and use.
8. **Troubleshooting** — common errors and fixes (API key missing, rate limits, etc.).
9. **Development** — running tests, project structure.

### 5.5.3 — Error Messages

Improve all error messages to be user-friendly:

```python
# Bad:
raise ValueError(f"Planner failed after {attempt} attempts: {e}")

# Good:
raise ValueError(
    f"Failed to create book plan after {attempt} attempts.\n"
    f"This is usually caused by:\n"
    f"  1. Groq API rate limits — wait 60 seconds and try again\n"
    f"  2. The book topic is too vague — try a more specific brief\n"
    f"  3. Network issues — check your internet connection\n"
    f"Original error: {e}"
)
```

### 5.5.4 — Example Runs

Create `examples/` directory with:

```
examples/
├── example_brief_nonfiction.txt    # "A beginner's guide to personal finance"
├── example_brief_fiction.txt       # "A short mystery set in a small coastal town"
├── example_brief_selfhelp.txt      # "Overcoming procrastination for creative professionals"
├── example_config.yaml             # A config with all options documented
└── README.md                       # How to run the examples
```

---

## 5.6 — Verification Checklist

After completing Phase 5, verify:

### Human-in-Loop
- [ ] With `human_review.enabled: true`, the system pauses after Book Bible creation.
- [ ] The user can approve, request changes, or reject the Book Bible.
- [ ] Requesting changes re-runs the Planner with user feedback.
- [ ] Rejecting stops the pipeline cleanly.
- [ ] With `approve_chapters: true`, each chapter is shown for approval.
- [ ] `--resume` correctly resumes after the user closes the terminal at a checkpoint.
- [ ] With `human_review.enabled: false` (default), no pauses occur.

### PDF Export
- [ ] `output/book.pdf` is generated and opens correctly in a PDF reader.
- [ ] The PDF has a title page, table of contents, and chapters.
- [ ] Images are embedded in the PDF where available.
- [ ] The PDF is readable (reasonable font sizes, margins, line spacing).

### Genre Presets
- [ ] Selecting "fiction" produces a noticeably different Book Bible than "non-fiction how-to".
- [ ] Genre presets affect tone, style, and chapter structure.
- [ ] An unrecognized genre falls back to generic presets without crashing.

### Test Suite
- [ ] `pytest tests/ -v` passes all tests.
- [ ] Tests run without any API keys or network access.
- [ ] Coverage is at least 70% for `models/`, `utils/`, and `orchestrator/`.
- [ ] The golden run test validates expected output structure.

### Documentation
- [ ] README.md covers installation, configuration, usage, and troubleshooting.
- [ ] A new user can set up and run the system from the README alone.
- [ ] Example briefs are provided and runnable.

### Overall
- [ ] The full pipeline completes in both autonomous and human-supervised modes.
- [ ] All three export formats (MD, DOCX, PDF) are produced.
- [ ] Error messages are clear and actionable for common failure cases.
- [ ] The system handles graceful degradation at every optional stage (images, research, trends, PDF).
