"""Image-related models."""

from pydantic import BaseModel


class ImagePrompt(BaseModel):
    """Prompt for future image generation."""

    anchor_id: str = ""
    chapter_number: int
    position: str = ""
    prompt: str
    style: str = ""
    alt_text: str = ""


class ImageAnchor(BaseModel):
    """Exact image marker parsed from accepted chapter Markdown."""

    anchor_id: str
    chapter_number: int
    marker: str
    nearest_heading: str = ""
    context_before: str = ""
    context_after: str = ""


class ImageAsset(BaseModel):
    """Generated or placeholder image asset."""

    anchor_id: str = ""
    chapter_number: int
    position: str = ""
    file_path: str = ""
    prompt_used: str = ""
    alt_text: str = ""
    is_placeholder: bool = True
