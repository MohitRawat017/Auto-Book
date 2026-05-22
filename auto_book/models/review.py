"""Review decision models."""

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
    score: float = Field(ge=0, le=10)
    problems: list[str] = Field(default_factory=list)
    required_fixes: list[str] = Field(default_factory=list)
    suggested_edits: list[str] = Field(default_factory=list)
    continuity_issues: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    tone_match: bool = True
