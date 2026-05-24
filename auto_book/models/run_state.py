"""Run state models for checkpointing."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.export import ExportResult
from auto_book.models.image import ImageAsset
from auto_book.models.memory import DynamicMemory
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
    RATE_LIMITED = "rate_limited"
    COMPLETED = "completed"
    FAILED = "failed"


class ChapterStatus(BaseModel):
    """Tracks a chapter through the pipeline."""

    chapter_number: int
    status: str = "pending"
    revision_count: int = 0
    regeneration_count: int = 0
    review_passes: int = 0
    current_draft: ChapterDraft | None = None
    last_review: ReviewDecision | None = None
    accepted_draft: ChapterDraft | None = None


class TokenUsage(BaseModel):
    """Cumulative token usage for later live agent phases."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls_made: int = 0


class RunState(BaseModel):
    """Complete checkpoint of a book generation run."""

    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    user_brief: str
    genre: str = "non-fiction how-to"
    phase: RunPhase = RunPhase.INITIALIZED
    current_chapter: int = 0
    book_bible: BookBible | None = None
    dynamic_memory: DynamicMemory = Field(default_factory=DynamicMemory)
    chapter_statuses: list[ChapterStatus] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    image_assets: list[ImageAsset] = Field(default_factory=list)
    cover_asset: ImageAsset | None = None
    export_result: ExportResult | None = None
    failure_reason: str = ""
    retry_after_seconds: int = 0
    resume_not_before: datetime | None = None
    output_directory: str = "./output"
