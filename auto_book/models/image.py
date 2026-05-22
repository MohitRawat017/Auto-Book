"""Image-related models."""

from pydantic import BaseModel


class ImagePrompt(BaseModel):
    """Prompt for future image generation."""

    chapter_number: int
    position: str
    prompt: str
    style: str = ""
    alt_text: str = ""


class ImageAsset(BaseModel):
    """Generated or placeholder image asset."""

    chapter_number: int
    position: str
    file_path: str = ""
    prompt_used: str = ""
    alt_text: str = ""
    is_placeholder: bool = True
