"""Dynamic memory models."""

from pydantic import BaseModel, Field


class ChapterMemoryEntry(BaseModel):
    """Summary of an accepted chapter for continuity."""

    chapter_number: int
    title: str
    summary: str
    key_claims: list[str] = Field(default_factory=list)
    definitions_introduced: list[str] = Field(default_factory=list)
    characters_or_concepts: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    research_facts_used: list[str] = Field(default_factory=list)


class DynamicMemory(BaseModel):
    """Rolling memory updated after every accepted chapter."""

    chapters: list[ChapterMemoryEntry] = Field(default_factory=list)
    recurring_terms: dict[str, str] = Field(default_factory=dict)
    style_observations: list[str] = Field(default_factory=list)
    repetition_warnings: list[str] = Field(default_factory=list)
    total_word_count: int = 0
    token_count_estimate: int = 0
