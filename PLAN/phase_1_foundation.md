# Phase 1 — Foundation & Skeleton

## Goal

Set up the complete project scaffolding: directory structure, dependencies, configuration system, all Pydantic data models, a runnable LangGraph orchestrator skeleton with stub agents, run-state checkpointing, structured logging, and a CLI entry point.

**Success Criteria**: Running `python -m auto_book "Write a book about X"` walks through every orchestrator node, prints stub outputs, and saves/loads run state to disk at each step.

---

## 1.1 — Project Structure

Create the following directory layout inside `d:\vs code\Auto_Book`:

```
auto_book/
├── __init__.py
├── __main__.py              # CLI entry point
├── cli.py                   # Argument parsing and user input
├── config.py                # Config loading (.env + config.yaml)
├── models/
│   ├── __init__.py
│   ├── book_bible.py        # BookBible Pydantic model
│   ├── chapter.py           # ChapterPlan, ChapterDraft models
│   ├── review.py            # ReviewDecision model
│   ├── memory.py            # DynamicMemory model
│   ├── run_state.py         # RunState model (checkpointing)
│   ├── image.py             # ImagePlan, ImageAsset models
│   └── export.py            # ExportResult model
├── agents/
│   ├── __init__.py
│   ├── planner.py           # Planner Agent (stub in Phase 1)
│   ├── writer.py            # Writer Agent (stub in Phase 1)
│   ├── reviewer.py          # Reviewer Agent (stub in Phase 1)
│   ├── memory_updater.py    # Memory Update Agent (stub in Phase 1)
│   ├── image_agent.py       # Image Agent (stub in Phase 1)
│   └── assembler.py         # Assembler (stub in Phase 1)
├── orchestrator/
│   ├── __init__.py
│   ├── graph.py             # LangGraph state machine definition
│   ├── state.py             # Graph state TypedDict
│   └── checkpointer.py     # Save/load run state to disk
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Structured logging setup
│   ├── tokens.py            # Token counting and budget helpers
│   └── validation.py        # Semantic validation helpers
├── output/                   # Default output directory (gitignored)
config.yaml                   # Default project config
.env.example                  # Example .env with required keys
```

### Implementation Steps

1. Create every directory and `__init__.py` file listed above.
2. Add `output/` to `.gitignore`.
3. Create empty placeholder files for every module.

---

## 1.2 — Dependencies

Update `pyproject.toml` to add all Phase 1 dependencies.

### Required packages

```toml
[project]
name = "auto-book"
version = "0.1.0"
description = "Autonomous book-writing agent pipeline"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.4.0",
    "langchain>=0.3.0",
    "langchain-groq>=0.3.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "tiktoken>=0.9.0",
    "python-dotenv>=1.0",
    "rich>=13.0",
]

[project.scripts]
auto-book = "auto_book.__main__:main"
```

**Package purposes:**
- `langgraph` — orchestrator state machine
- `langchain` + `langchain-groq` — LLM interface to Groq API
- `pydantic` + `pydantic-settings` — data models and .env config loading
- `pyyaml` — config.yaml loading
- `tiktoken` — token counting for context budget management
- `python-dotenv` — .env file loading
- `rich` — pretty CLI output and progress display

### Install command

```bash
uv sync
```

---

## 1.3 — Configuration System

### 1.3.1 — `.env` file (secrets)

Create `.env.example` with:

```env
GROQ_API_KEY=your_groq_api_key_here
KIE_API_KEY=your_kie_ai_api_key_here
```

The actual `.env` file should be in `.gitignore`. The application loads it at startup.

### 1.3.2 — `config.yaml` (project settings)

Create `config.yaml` with sensible defaults:

```yaml
# Auto_Book Configuration

# LLM Settings
llm:
  provider: "groq"
  writer_model: "meta-llama/llama-4-maverick-17b-128e-instruct"
  reviewer_model: "meta-llama/llama-4-scout-17b-16e-instruct"
  planner_model: "meta-llama/llama-4-maverick-17b-128e-instruct"
  memory_model: "meta-llama/llama-4-scout-17b-16e-instruct"
  temperature_creative: 0.7    # For writer/planner
  temperature_analytical: 0.2  # For reviewer/memory

# Book Defaults
book:
  default_chapter_count: 10
  target_words_per_chapter: 1500
  min_words_per_chapter: 800
  max_words_per_chapter: 2500
  target_total_pages: 30

# Retry Settings
retry:
  max_revisions: 2          # Max targeted revisions per chapter
  max_regenerations: 1      # Max full regenerations per chapter
  max_validation_retries: 3 # Max retries for malformed LLM output
  retry_delay_seconds: 2    # Delay between retries (rate limit safety)

# Context Window Budget (in tokens)
context_budget:
  system_prompt: 500
  book_bible: 1500
  dynamic_memory: 1500
  research_notes: 1000       # Reserved for Phase 4
  chapter_plan: 500
  generation_headroom: 3000
  total_max: 8000

# Output Settings
output:
  directory: "./output"
  save_intermediate: true    # Save drafts, reviews, memory snapshots

# Logging
logging:
  level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_file: "./output/run.log"
```

### 1.3.3 — `config.py` implementation

```python
"""
config.py — Load and validate configuration from .env and config.yaml

This module exposes two objects:
  - `secrets`: loaded from .env (API keys)
  - `settings`: loaded from config.yaml (project settings)

Usage:
    from auto_book.config import secrets, settings
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel
import yaml


class Secrets(BaseSettings):
    """Loaded from .env file. Contains API keys only."""
    groq_api_key: str
    kie_api_key: str = ""  # Optional for Phase 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class LLMSettings(BaseModel):
    provider: str = "groq"
    writer_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    reviewer_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    planner_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    memory_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature_creative: float = 0.7
    temperature_analytical: float = 0.2


class BookSettings(BaseModel):
    default_chapter_count: int = 10
    target_words_per_chapter: int = 1500
    min_words_per_chapter: int = 800
    max_words_per_chapter: int = 2500
    target_total_pages: int = 30


class RetrySettings(BaseModel):
    max_revisions: int = 2
    max_regenerations: int = 1
    max_validation_retries: int = 3
    retry_delay_seconds: int = 2


class ContextBudget(BaseModel):
    system_prompt: int = 500
    book_bible: int = 1500
    dynamic_memory: int = 1500
    research_notes: int = 1000
    chapter_plan: int = 500
    generation_headroom: int = 3000
    total_max: int = 8000


class OutputSettings(BaseModel):
    directory: str = "./output"
    save_intermediate: bool = True


class LoggingSettings(BaseModel):
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "./output/run.log"


class Settings(BaseModel):
    """All non-secret settings loaded from config.yaml."""
    llm: LLMSettings = LLMSettings()
    book: BookSettings = BookSettings()
    retry: RetrySettings = RetrySettings()
    context_budget: ContextBudget = ContextBudget()
    output: OutputSettings = OutputSettings()
    logging: LoggingSettings = LoggingSettings()


def load_settings(config_path: str = "config.yaml") -> Settings:
    """Load settings from config.yaml. Falls back to defaults if file is missing."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Settings(**raw)
    return Settings()


# Module-level singletons — import these directly
secrets = Secrets()
settings = load_settings()
```

---

## 1.4 — Pydantic Data Models

Every model below should be defined in its own file under `auto_book/models/`. Each model uses Pydantic v2 `BaseModel`. All fields should have type hints and docstrings.

### 1.4.1 — `models/book_bible.py`

```python
"""Book Bible — the stable source of truth for the entire book run."""

from pydantic import BaseModel, Field


class ChapterOutline(BaseModel):
    """Outline for a single chapter."""
    chapter_number: int
    title: str
    summary: str = Field(description="2-3 sentence summary of what this chapter covers")
    key_topics: list[str] = Field(description="Main topics/concepts this chapter must address")
    word_count_target: int = 1500
    depends_on: list[int] = Field(
        default_factory=list,
        description="Chapter numbers this chapter builds upon"
    )


class BookBible(BaseModel):
    """
    The 'soul' of the book. Created by the Planner Agent.
    This is the stable reference document used by Writer and Reviewer.
    """
    working_title: str
    subtitle: str = ""
    genre: str = Field(description="e.g. 'non-fiction how-to', 'fiction thriller', 'self-help'")
    core_thesis: str = Field(description="The central argument or narrative premise of the book")
    target_audience: str = Field(description="Who this book is written for")
    reader_pain_points: list[str] = Field(
        default_factory=list,
        description="Problems or desires the reader has that this book addresses"
    )
    book_promise: str = Field(description="What the reader will gain from reading this book")
    tone: str = Field(description="e.g. 'conversational and warm', 'academic but accessible'")
    style_guide: str = Field(description="Writing style rules: sentence length, vocabulary level, etc.")
    chapter_outline: list[ChapterOutline]
    required_themes: list[str] = Field(default_factory=list)
    forbidden_topics: list[str] = Field(
        default_factory=list,
        description="Topics or claims to explicitly avoid"
    )
    glossary: dict[str, str] = Field(
        default_factory=dict,
        description="Key terms and their definitions for consistency"
    )
    image_direction: str = Field(
        default="",
        description="General guidance for image style if images are used"
    )
```

### 1.4.2 — `models/chapter.py`

```python
"""Chapter-related models: plan and draft."""

from pydantic import BaseModel, Field


class ChapterPlan(BaseModel):
    """The plan for writing a specific chapter — extracted from BookBible for the Writer."""
    chapter_number: int
    title: str
    summary: str
    key_topics: list[str]
    word_count_target: int
    continuity_notes: str = Field(
        default="",
        description="Notes on what previous chapters established that this chapter should reference"
    )


class ImagePlaceholder(BaseModel):
    """A placeholder for where an image should go in a chapter."""
    position: str = Field(description="e.g. 'after paragraph 3', 'chapter header'")
    description: str = Field(description="What the image should depict")
    alt_text: str = ""


class ChapterDraft(BaseModel):
    """A single chapter's draft output from the Writer Agent."""
    chapter_number: int
    title: str
    body: str = Field(description="The full chapter text in Markdown format")
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Key points from this chapter (for non-fiction)"
    )
    image_placeholders: list[ImagePlaceholder] = Field(default_factory=list)
    word_count: int = 0
    sources_referenced: list[str] = Field(
        default_factory=list,
        description="Research sources used in this chapter"
    )
```

### 1.4.3 — `models/review.py`

```python
"""Review decision model returned by the Reviewer Agent."""

from enum import Enum
from pydantic import BaseModel, Field


class ReviewDecisionEnum(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    FAIL = "fail"


class ReviewDecision(BaseModel):
    """Structured review output for a chapter draft."""
    chapter_number: int
    decision: ReviewDecisionEnum
    score: float = Field(ge=0, le=10, description="Quality score from 0-10")
    problems: list[str] = Field(
        default_factory=list,
        description="Specific problems identified"
    )
    required_fixes: list[str] = Field(
        default_factory=list,
        description="Concrete fixes the Writer must make for a revision"
    )
    suggested_edits: list[str] = Field(
        default_factory=list,
        description="Optional improvements that would enhance quality"
    )
    continuity_issues: list[str] = Field(
        default_factory=list,
        description="Conflicts with previous chapters or Book Bible"
    )
    research_gaps: list[str] = Field(
        default_factory=list,
        description="Claims that lack supporting evidence"
    )
    tone_match: bool = Field(
        default=True,
        description="Whether the chapter matches the Book Bible tone"
    )
```

### 1.4.4 — `models/memory.py`

```python
"""Dynamic Memory — evolves after every accepted chapter."""

from pydantic import BaseModel, Field


class ChapterMemoryEntry(BaseModel):
    """Summary of an accepted chapter stored in Dynamic Memory."""
    chapter_number: int
    title: str
    summary: str = Field(description="3-5 sentence summary of this chapter's content")
    key_claims: list[str] = Field(
        default_factory=list,
        description="Important claims or arguments made in this chapter"
    )
    definitions_introduced: list[str] = Field(default_factory=list)
    characters_or_concepts: list[str] = Field(
        default_factory=list,
        description="Named entities, characters, or key concepts introduced"
    )
    open_loops: list[str] = Field(
        default_factory=list,
        description="Questions, promises, or threads left unresolved for later chapters"
    )
    research_facts_used: list[str] = Field(default_factory=list)


class DynamicMemory(BaseModel):
    """
    The rolling memory of the book's progress.
    Updated after each accepted chapter.
    Used by the Writer for continuity.
    """
    chapters: list[ChapterMemoryEntry] = Field(default_factory=list)
    recurring_terms: dict[str, str] = Field(
        default_factory=dict,
        description="Terms used consistently across chapters -> their definition"
    )
    style_observations: list[str] = Field(
        default_factory=list,
        description="Notes about writing style patterns observed so far"
    )
    repetition_warnings: list[str] = Field(
        default_factory=list,
        description="Topics/points that have been covered and should not be repeated"
    )
    total_word_count: int = 0
    token_count_estimate: int = Field(
        default=0,
        description="Estimated token size of this memory object for budget tracking"
    )
```

### 1.4.5 — `models/run_state.py`

```python
"""Run State — the complete checkpoint of a book generation run."""

from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from auto_book.models.book_bible import BookBible
from auto_book.models.memory import DynamicMemory
from auto_book.models.chapter import ChapterDraft
from auto_book.models.review import ReviewDecision


class RunPhase(str, Enum):
    INITIALIZED = "initialized"
    PLANNING = "planning"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    MEMORY_UPDATE = "memory_update"
    IMAGE_GENERATION = "image_generation"
    ASSEMBLING = "assembling"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class ChapterStatus(BaseModel):
    """Tracks the status of a single chapter through the pipeline."""
    chapter_number: int
    status: str = "pending"  # pending, drafting, reviewing, accepted, blocked
    revision_count: int = 0
    regeneration_count: int = 0
    current_draft: ChapterDraft | None = None
    last_review: ReviewDecision | None = None
    accepted_draft: ChapterDraft | None = None


class TokenUsage(BaseModel):
    """Tracks cumulative token usage for cost awareness."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls_made: int = 0


class RunState(BaseModel):
    """
    Complete snapshot of a book generation run.
    Saved to disk after every major step for checkpointing/recovery.
    """
    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    user_brief: str = Field(description="The original topic/brief from the user")
    phase: RunPhase = RunPhase.INITIALIZED
    current_chapter: int = 0
    book_bible: BookBible | None = None
    dynamic_memory: DynamicMemory = Field(default_factory=DynamicMemory)
    chapter_statuses: list[ChapterStatus] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    failure_reason: str = ""
    output_directory: str = "./output"
```

### 1.4.6 — `models/image.py`

```python
"""Image-related models for the Image Agent."""

from pydantic import BaseModel, Field


class ImagePrompt(BaseModel):
    """A generated prompt for image creation."""
    chapter_number: int
    position: str
    prompt: str = Field(description="The image generation prompt")
    style: str = Field(default="", description="Style guidance (e.g. 'watercolor', 'minimalist')")
    alt_text: str = ""


class ImageAsset(BaseModel):
    """A generated or placeholder image asset."""
    chapter_number: int
    position: str
    file_path: str = ""  # Path to saved image file, empty if placeholder
    prompt_used: str = ""
    alt_text: str = ""
    is_placeholder: bool = True
```

### 1.4.7 — `models/export.py`

```python
"""Export result model."""

from pydantic import BaseModel, Field


class ExportResult(BaseModel):
    """Result of the final book export."""
    markdown_path: str = ""
    docx_path: str = ""
    pdf_path: str = ""  # Empty until Phase 5
    total_chapters: int = 0
    total_word_count: int = 0
    images_embedded: int = 0
    success: bool = True
    errors: list[str] = Field(default_factory=list)
```

### 1.4.8 — `models/__init__.py`

```python
"""Re-export all models for convenient importing."""

from auto_book.models.book_bible import BookBible, ChapterOutline
from auto_book.models.chapter import ChapterPlan, ChapterDraft, ImagePlaceholder
from auto_book.models.review import ReviewDecision, ReviewDecisionEnum
from auto_book.models.memory import DynamicMemory, ChapterMemoryEntry
from auto_book.models.run_state import RunState, RunPhase, ChapterStatus, TokenUsage
from auto_book.models.image import ImagePrompt, ImageAsset
from auto_book.models.export import ExportResult

__all__ = [
    "BookBible", "ChapterOutline",
    "ChapterPlan", "ChapterDraft", "ImagePlaceholder",
    "ReviewDecision", "ReviewDecisionEnum",
    "DynamicMemory", "ChapterMemoryEntry",
    "RunState", "RunPhase", "ChapterStatus", "TokenUsage",
    "ImagePrompt", "ImageAsset",
    "ExportResult",
]
```

---

## 1.5 — Utility Modules

### 1.5.1 — `utils/logger.py`

Set up structured logging using Python's `logging` module with JSON-formatted output.

```python
"""Structured logging for Auto_Book."""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: str = "./output/run.log"
) -> logging.Logger:
    """
    Create and configure the application logger.
    Call once at startup from __main__.py.
    """
    logger = logging.getLogger("auto_book")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    # Console handler — human-readable
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(module)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # File handler — JSON lines
    if log_to_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the application logger. Must call setup_logger() first."""
    return logging.getLogger("auto_book")
```

### 1.5.2 — `utils/tokens.py`

```python
"""Token counting and context budget helpers."""

import tiktoken


# Use cl100k_base as a reasonable approximation for most models
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(_ENCODING.encode(text))


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within a token budget.
    Cuts from the beginning (keeps most recent content).
    Returns the truncated string.
    """
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[-max_tokens:]
    return _ENCODING.decode(truncated_tokens)


def fits_budget(text: str, max_tokens: int) -> bool:
    """Check if text fits within a token budget."""
    return count_tokens(text) <= max_tokens
```

### 1.5.3 — `utils/validation.py`

```python
"""Semantic validation helpers beyond Pydantic schema validation."""

from auto_book.models.chapter import ChapterDraft
from auto_book.models.review import ReviewDecision, ReviewDecisionEnum


def validate_chapter_draft(draft: ChapterDraft, min_words: int, max_words: int) -> list[str]:
    """
    Validate a chapter draft beyond schema correctness.
    Returns a list of error strings. Empty list means valid.
    """
    errors = []
    word_count = len(draft.body.split())
    draft.word_count = word_count  # Update the word count field

    if word_count < min_words:
        errors.append(
            f"Chapter {draft.chapter_number} has {word_count} words, "
            f"minimum is {min_words}"
        )
    if word_count > max_words:
        errors.append(
            f"Chapter {draft.chapter_number} has {word_count} words, "
            f"maximum is {max_words}"
        )
    if not draft.title.strip():
        errors.append(f"Chapter {draft.chapter_number} has an empty title")
    if not draft.body.strip():
        errors.append(f"Chapter {draft.chapter_number} has an empty body")

    return errors


def validate_review_decision(review: ReviewDecision) -> list[str]:
    """
    Validate a review decision beyond schema correctness.
    Returns a list of error strings.
    """
    errors = []

    if review.decision == ReviewDecisionEnum.REVISE and not review.required_fixes:
        errors.append(
            f"Review for chapter {review.chapter_number} says 'revise' "
            f"but provides no required_fixes"
        )
    if review.decision == ReviewDecisionEnum.FAIL and not review.problems:
        errors.append(
            f"Review for chapter {review.chapter_number} says 'fail' "
            f"but lists no problems"
        )

    return errors
```

---

## 1.6 — Orchestrator Skeleton (LangGraph)

### 1.6.1 — `orchestrator/state.py` — Graph State Definition

```python
"""
LangGraph state definition.

This TypedDict defines ALL data that flows through the graph.
Every node reads from and writes to this shared state.
"""

from typing import TypedDict
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft, ChapterPlan
from auto_book.models.review import ReviewDecision
from auto_book.models.memory import DynamicMemory
from auto_book.models.run_state import RunPhase, ChapterStatus, TokenUsage
from auto_book.models.image import ImageAsset
from auto_book.models.export import ExportResult


class GraphState(TypedDict, total=False):
    """The shared state object passed through all LangGraph nodes."""

    # --- Input ---
    user_brief: str
    genre: str

    # --- Planning outputs ---
    book_bible: BookBible | None

    # --- Chapter loop state ---
    current_chapter: int
    chapter_plan: ChapterPlan | None
    chapter_draft: ChapterDraft | None
    review_decision: ReviewDecision | None
    revision_count: int
    regeneration_count: int

    # --- Memory ---
    dynamic_memory: DynamicMemory

    # --- Tracking ---
    chapter_statuses: list[ChapterStatus]
    phase: RunPhase
    token_usage: TokenUsage

    # --- Image ---
    image_assets: list[ImageAsset]

    # --- Export ---
    export_result: ExportResult | None

    # --- Error tracking ---
    error: str
```

### 1.6.2 — `orchestrator/graph.py` — LangGraph State Machine

This is the core file. It defines the graph structure with nodes and edges.

**In Phase 1, every agent node is a stub** that prints what it would do and returns dummy data. The graph structure, routing logic, and conditional edges are all real.

```python
"""
LangGraph orchestrator — the main state machine.

Nodes:
  1. collect_input    — Gather user brief and genre
  2. plan_book        — Planner Agent creates Book Bible
  3. prepare_chapter  — Extract chapter plan from Book Bible
  4. write_chapter    — Writer Agent drafts a chapter
  5. review_chapter   — Reviewer Agent evaluates the draft
  6. update_memory    — Update Dynamic Memory after accepted chapter
  7. check_next       — Decide: next chapter, or assembly?
  8. assemble_book    — Assembler merges everything
  9. export_book      — Export to Markdown + DOCX
  10. handle_failure  — Handle unrecoverable errors

Conditional edges:
  - After review: route to update_memory (pass), write_chapter (revise),
    write_chapter with stronger constraints (fail), or handle_failure (retry limit).
  - After check_next: route to prepare_chapter (more chapters) or assemble_book (done).
"""

from langgraph.graph import StateGraph, END
from auto_book.orchestrator.state import GraphState
from auto_book.models.run_state import RunPhase
from auto_book.models.review import ReviewDecisionEnum
from auto_book.utils.logger import get_logger


# ─── Node Functions (STUBS in Phase 1) ───

def collect_input(state: GraphState) -> dict:
    """Node 1: Validate and prepare user input."""
    logger = get_logger()
    logger.info(f"[collect_input] Brief: {state.get('user_brief', '')[:80]}...")
    return {"phase": RunPhase.INITIALIZED}


def plan_book(state: GraphState) -> dict:
    """Node 2: STUB — Planner Agent creates Book Bible."""
    logger = get_logger()
    logger.info("[plan_book] STUB — would call Planner Agent here")
    # Phase 2 replaces this with real Planner Agent
    return {"phase": RunPhase.PLANNING, "book_bible": None}


def prepare_chapter(state: GraphState) -> dict:
    """Node 3: Extract the current chapter's plan from the Book Bible."""
    logger = get_logger()
    current = state.get("current_chapter", 0) + 1
    logger.info(f"[prepare_chapter] Preparing chapter {current}")
    return {
        "current_chapter": current,
        "chapter_plan": None,
        "revision_count": 0,
        "regeneration_count": 0,
        "phase": RunPhase.WRITING,
    }


def write_chapter(state: GraphState) -> dict:
    """Node 4: STUB — Writer Agent drafts a chapter."""
    logger = get_logger()
    ch = state.get("current_chapter", 0)
    logger.info(f"[write_chapter] STUB — would write chapter {ch}")
    return {"chapter_draft": None, "phase": RunPhase.WRITING}


def review_chapter(state: GraphState) -> dict:
    """Node 5: STUB — Reviewer Agent evaluates the draft."""
    logger = get_logger()
    ch = state.get("current_chapter", 0)
    logger.info(f"[review_chapter] STUB — would review chapter {ch}")
    return {"review_decision": None, "phase": RunPhase.REVIEWING}


def update_memory(state: GraphState) -> dict:
    """Node 6: STUB — Update Dynamic Memory after accepted chapter."""
    logger = get_logger()
    ch = state.get("current_chapter", 0)
    logger.info(f"[update_memory] STUB — would update memory for chapter {ch}")
    return {"phase": RunPhase.MEMORY_UPDATE}


def check_next(state: GraphState) -> dict:
    """Node 7: Decide if there are more chapters or we should assemble."""
    logger = get_logger()
    current = state.get("current_chapter", 0)
    bible = state.get("book_bible")
    total = len(bible.chapter_outline) if bible else 10  # default in stub mode
    logger.info(f"[check_next] Chapter {current}/{total}")
    return {}


def assemble_book(state: GraphState) -> dict:
    """Node 8: STUB — Assembler merges all chapters."""
    logger = get_logger()
    logger.info("[assemble_book] STUB — would assemble final book")
    return {"phase": RunPhase.ASSEMBLING}


def export_book(state: GraphState) -> dict:
    """Node 9: STUB — Export to Markdown + DOCX."""
    logger = get_logger()
    logger.info("[export_book] STUB — would export book")
    return {"phase": RunPhase.COMPLETED, "export_result": None}


def handle_failure(state: GraphState) -> dict:
    """Node 10: Handle unrecoverable failure."""
    logger = get_logger()
    error = state.get("error", "Unknown error")
    logger.error(f"[handle_failure] Run failed: {error}")
    return {"phase": RunPhase.FAILED}


# ─── Routing Functions ───

def route_after_review(state: GraphState) -> str:
    """
    Conditional edge after review_chapter.
    Routes based on review decision + retry counts.
    """
    review = state.get("review_decision")

    # In stub mode (Phase 1), simulate a pass
    if review is None:
        return "update_memory"

    revision_count = state.get("revision_count", 0)
    regeneration_count = state.get("regeneration_count", 0)

    if review.decision == ReviewDecisionEnum.PASS:
        return "update_memory"
    elif review.decision == ReviewDecisionEnum.REVISE:
        if revision_count < 2:  # max_revisions from config
            return "write_chapter"
        else:
            return "handle_failure"
    elif review.decision == ReviewDecisionEnum.FAIL:
        if regeneration_count < 1:  # max_regenerations from config
            return "write_chapter"
        else:
            return "handle_failure"

    return "handle_failure"


def route_after_check(state: GraphState) -> str:
    """
    Conditional edge after check_next.
    Routes to next chapter or assembly.
    """
    current = state.get("current_chapter", 0)
    bible = state.get("book_bible")
    total = len(bible.chapter_outline) if bible else 10

    if current < total:
        return "prepare_chapter"
    else:
        return "assemble_book"


# ─── Build the Graph ───

def build_graph() -> StateGraph:
    """
    Construct the LangGraph state machine.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("collect_input", collect_input)
    graph.add_node("plan_book", plan_book)
    graph.add_node("prepare_chapter", prepare_chapter)
    graph.add_node("write_chapter", write_chapter)
    graph.add_node("review_chapter", review_chapter)
    graph.add_node("update_memory", update_memory)
    graph.add_node("check_next", check_next)
    graph.add_node("assemble_book", assemble_book)
    graph.add_node("export_book", export_book)
    graph.add_node("handle_failure", handle_failure)

    # Linear edges
    graph.set_entry_point("collect_input")
    graph.add_edge("collect_input", "plan_book")
    graph.add_edge("plan_book", "prepare_chapter")
    graph.add_edge("write_chapter", "review_chapter")
    graph.add_edge("update_memory", "check_next")
    graph.add_edge("assemble_book", "export_book")
    graph.add_edge("export_book", END)
    graph.add_edge("handle_failure", END)

    # Conditional edges
    graph.add_conditional_edges(
        "review_chapter",
        route_after_review,
        {
            "update_memory": "update_memory",
            "write_chapter": "write_chapter",
            "handle_failure": "handle_failure",
        }
    )
    graph.add_conditional_edges(
        "check_next",
        route_after_check,
        {
            "prepare_chapter": "prepare_chapter",
            "assemble_book": "assemble_book",
        }
    )

    # Edge from prepare_chapter to write_chapter
    graph.add_edge("prepare_chapter", "write_chapter")

    return graph.compile()
```

### 1.6.3 — `orchestrator/checkpointer.py` — Save/Load Run State

```python
"""
Checkpointer — save and load run state to/from disk.

After every major step, the orchestrator saves a RunState snapshot.
On restart, it can resume from the last checkpoint.
"""

import json
from pathlib import Path
from datetime import datetime
from auto_book.models.run_state import RunState
from auto_book.utils.logger import get_logger


CHECKPOINT_FILENAME = "run_state.json"


def save_checkpoint(state: RunState, output_dir: str) -> None:
    """Save the current run state to disk as JSON."""
    logger = get_logger()
    path = Path(output_dir) / CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)

    state.updated_at = datetime.now()
    data = state.model_dump_json(indent=2)

    path.write_text(data, encoding="utf-8")
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(output_dir: str) -> RunState | None:
    """
    Load a run state from disk.
    Returns None if no checkpoint exists.
    """
    logger = get_logger()
    path = Path(output_dir) / CHECKPOINT_FILENAME

    if not path.exists():
        logger.info("No existing checkpoint found — starting fresh")
        return None

    try:
        data = path.read_text(encoding="utf-8")
        state = RunState.model_validate_json(data)
        logger.info(
            f"Checkpoint loaded — Phase: {state.phase.value}, "
            f"Chapter: {state.current_chapter}"
        )
        return state
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}. Starting fresh.")
        return None
```

---

## 1.7 — CLI Entry Point

### 1.7.1 — `cli.py`

```python
"""CLI argument parsing and user interaction."""

import argparse
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="auto-book",
        description="Autonomous Book Writing Agent"
    )
    parser.add_argument(
        "brief",
        nargs="?",
        default=None,
        help="The book topic or brief (e.g., 'A beginner guide to investing')"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (overrides config.yaml)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint"
    )
    return parser.parse_args()


def get_user_brief() -> str:
    """Interactively ask the user for a book brief if not provided as argument."""
    console.print("\n[bold cyan]╔══════════════════════════════════════╗[/]")
    console.print("[bold cyan]║     Auto_Book — Book Generator       ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════╝[/]\n")

    brief = Prompt.ask(
        "[bold]What should the book be about?[/]\n"
        "Describe the topic, target audience, and any specific requirements"
    )
    return brief


def get_genre() -> str:
    """Ask the user what genre/type of book they want."""
    genre = Prompt.ask(
        "\n[bold]What genre or type of book?[/]",
        choices=[
            "non-fiction how-to",
            "non-fiction informational",
            "self-help",
            "fiction",
            "memoir",
            "other",
        ],
        default="non-fiction how-to"
    )
    if genre == "other":
        genre = Prompt.ask("Describe the genre")
    return genre
```

### 1.7.2 — `__main__.py`

```python
"""
Auto_Book entry point.

Usage:
    python -m auto_book "Write a book about investing for beginners"
    python -m auto_book --resume
    python -m auto_book
"""

import uuid
from auto_book.cli import parse_args, get_user_brief, get_genre
from auto_book.config import settings, load_settings
from auto_book.utils.logger import setup_logger, get_logger
from auto_book.orchestrator.graph import build_graph
from auto_book.orchestrator.checkpointer import load_checkpoint, save_checkpoint
from auto_book.models.run_state import RunState
from rich.console import Console

console = Console()


def main():
    args = parse_args()

    # Load config
    cfg = load_settings(args.config) if args.config else settings
    output_dir = args.output or cfg.output.directory

    # Setup logging
    setup_logger(
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_file=cfg.logging.log_file,
    )
    logger = get_logger()

    # Resume or start fresh
    if args.resume:
        run_state = load_checkpoint(output_dir)
        if run_state is None:
            console.print("[red]No checkpoint found to resume from.[/]")
            return
        brief = run_state.user_brief
        genre = run_state.book_bible.genre if run_state.book_bible else "non-fiction"
    else:
        brief = args.brief or get_user_brief()
        genre = get_genre()
        run_state = RunState(
            run_id=str(uuid.uuid4())[:8],
            user_brief=brief,
            output_directory=output_dir,
        )

    logger.info(f"Starting run {run_state.run_id}")
    logger.info(f"Brief: {brief[:100]}")
    logger.info(f"Genre: {genre}")

    # Build and run the graph
    graph = build_graph()

    initial_state = {
        "user_brief": brief,
        "genre": genre,
        "dynamic_memory": run_state.dynamic_memory,
        "current_chapter": run_state.current_chapter,
        "chapter_statuses": run_state.chapter_statuses,
        "phase": run_state.phase,
        "token_usage": run_state.token_usage,
        "revision_count": 0,
        "regeneration_count": 0,
        "image_assets": [],
        "error": "",
    }

    console.print(f"\n[bold green]▶ Starting book generation...[/]\n")

    try:
        result = graph.invoke(initial_state)
        console.print(f"\n[bold green]✓ Run completed — Phase: {result.get('phase')}[/]")
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Run interrupted by user[/]")
        logger.warning("Run interrupted by user")
    except Exception as e:
        console.print(f"\n[red]✗ Run failed: {e}[/]")
        logger.exception("Run failed with exception")


if __name__ == "__main__":
    main()
```

---

## 1.8 — Verification Checklist

After completing Phase 1, verify:

- [ ] `uv sync` installs all dependencies without errors.
- [ ] `python -m auto_book "Test topic"` runs and walks through all stub nodes.
- [ ] Console output shows each node being visited in order: `collect_input → plan_book → prepare_chapter → write_chapter → review_chapter → update_memory → check_next → ... → assemble_book → export_book`.
- [ ] A `run.log` file is created in the output directory with JSON-formatted log entries.
- [ ] The checkpoint file `run_state.json` can be written and read back.
- [ ] All Pydantic models can be instantiated with valid data and reject invalid data.
- [ ] The `config.yaml` loads correctly and all defaults are applied when fields are missing.
- [ ] `.env` loading works and raises a clear error if `GROQ_API_KEY` is missing.
- [ ] Token counting in `utils/tokens.py` returns reasonable counts for sample text.

### Quick Smoke Test Commands

```bash
# Install deps
uv sync

# Run with a topic
python -m auto_book "A beginner's guide to personal finance"

# Test config loading
python -c "from auto_book.config import settings; print(settings.llm.writer_model)"

# Test models
python -c "from auto_book.models import BookBible; print(BookBible.model_json_schema())"
```
