"""Chapter planning and draft models."""

from pydantic import BaseModel, Field


class ChapterPlan(BaseModel):
    """The plan used to write one chapter."""

    chapter_number: int
    title: str
    summary: str
    key_topics: list[str] = Field(default_factory=list)
    word_count_target: int = 1500
    continuity_notes: str = ""


class ImagePlaceholder(BaseModel):
    """Placeholder for a future image inside a chapter."""

    position: str
    description: str
    alt_text: str = ""


class ChapterDraft(BaseModel):
    """A single chapter draft."""

    chapter_number: int
    title: str
    body: str
    key_takeaways: list[str] = Field(default_factory=list)
    image_placeholders: list[ImagePlaceholder] = Field(default_factory=list)
    word_count: int = 0
    sources_referenced: list[str] = Field(default_factory=list)
