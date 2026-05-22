"""Export result model."""

from pydantic import BaseModel, Field


class ExportResult(BaseModel):
    """Result of a book export step."""

    markdown_path: str = ""
    docx_path: str = ""
    pdf_path: str = ""
    total_chapters: int = 0
    total_word_count: int = 0
    images_embedded: int = 0
    success: bool = True
    errors: list[str] = Field(default_factory=list)
