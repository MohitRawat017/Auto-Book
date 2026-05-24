"""LangGraph state definition."""

from datetime import datetime
from typing import TypedDict

from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft, ChapterPlan
from auto_book.models.export import ExportResult
from auto_book.models.image import ImageAsset
from auto_book.models.memory import DynamicMemory
from auto_book.models.review import ReviewDecision
from auto_book.models.run_state import ChapterStatus, RunPhase, TokenUsage


class GraphState(TypedDict, total=False):
    """Shared state object passed through LangGraph nodes."""

    run_id: str
    created_at: datetime
    updated_at: datetime
    user_brief: str
    genre: str
    output_directory: str
    book_bible: BookBible | None
    current_chapter: int
    chapter_plan: ChapterPlan | None
    chapter_draft: ChapterDraft | None
    review_decision: ReviewDecision | None
    revision_count: int
    regeneration_count: int
    review_passes: int
    dynamic_memory: DynamicMemory
    chapter_statuses: list[ChapterStatus]
    phase: RunPhase
    token_usage: TokenUsage
    image_assets: list[ImageAsset]
    cover_asset: ImageAsset | None
    export_result: ExportResult | None
    retry_after_seconds: int
    resume_not_before: datetime | None
    error: str
