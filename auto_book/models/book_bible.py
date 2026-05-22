"""Book Bible models."""

from pydantic import BaseModel, Field


class ChapterOutline(BaseModel):
    """Outline for a single chapter."""

    chapter_number: int
    title: str
    summary: str = Field(description="Two to three sentences describing the chapter.")
    key_topics: list[str] = Field(default_factory=list)
    word_count_target: int = 1500
    depends_on: list[int] = Field(default_factory=list)


class BookBible(BaseModel):
    """Stable source of truth for a book generation run."""

    working_title: str
    subtitle: str = ""
    genre: str
    core_thesis: str
    target_audience: str
    reader_pain_points: list[str] = Field(default_factory=list)
    book_promise: str
    tone: str
    style_guide: str
    chapter_outline: list[ChapterOutline]
    required_themes: list[str] = Field(default_factory=list)
    forbidden_topics: list[str] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    image_direction: str = ""
