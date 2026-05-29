"""Configuration loading for Auto_Book."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Optional secrets loaded from .env.

    Phase 1 is a stubbed skeleton and must run without API keys. Live agents in
    later phases should call the require_* helpers before making API requests.
    """

    groq_api_key: str = ""
    kie_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMSettings(BaseModel):
    provider: str = "groq"
    writer_model: str = "llama-3.3-70b-versatile"
    reviewer_model: str = "openai/gpt-oss-120b"
    planner_model: str = "qwen/qwen3-32b"
    memory_model: str = "openai/gpt-oss-20b"
    temperature_writer: float = 0.8
    temperature_planner: float = 0.7
    temperature_reviewer: float = 0.2
    temperature_memory: float = 0.1


class BookSettings(BaseModel):
    default_chapter_count: int = 10
    target_words_per_chapter: int = 1500
    min_words_per_chapter: int = 800
    max_words_per_chapter: int = 2500
    target_total_pages: int = 30


class RetrySettings(BaseModel):
    max_revisions: int = 3
    max_regenerations: int = 1
    max_validation_retries: int = 3
    retry_delay_seconds: int = 6


class ReviewSettings(BaseModel):
    enabled: bool = True
    max_review_passes: int = 1
    accept_score: float = 7.0
    soft_accept_score: float = 6.0


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


class ImageSettings(BaseModel):
    enabled: bool = True
    generate_actual: bool = True
    max_images_per_chapter: int = 3
    default_model: str = "qwen2/text-to-image"
    default_size: str = "16:9"
    output_format: str = "png"
    timeout_seconds: int = 120
    poll_interval_seconds: int = 3
    max_poll_attempts: int = 40
    fallback_to_placeholder: bool = True
    call_back_url: str = ""


class Settings(BaseModel):
    """All non-secret project settings loaded from config.yaml."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    book: BookSettings = Field(default_factory=BookSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    output: OutputSettings = Field(default_factory=OutputSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)


def load_settings(config_path: str = "config.yaml") -> Settings:
    """Load settings from config.yaml, falling back to defaults if missing."""

    path = Path(config_path)
    if not path.exists():
        return Settings()

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Settings(**raw)


def require_groq_api_key() -> str:
    """Return the Groq API key or raise a clear live-agent error."""

    if not secrets.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for live LLM agents. "
            "Set it in .env before running Phase 2+ functionality."
        )
    return secrets.groq_api_key


def set_settings(new_settings: Settings) -> None:
    """Replace module-level settings after loading a custom config file."""

    for field_name in Settings.model_fields:
        setattr(settings, field_name, getattr(new_settings, field_name))


secrets = Secrets()
settings = load_settings()
