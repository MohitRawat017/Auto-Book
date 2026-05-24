"""Save and load run checkpoints."""

from datetime import datetime
from pathlib import Path
from typing import Any

from auto_book.models.book_bible import BookBible
from auto_book.models.export import ExportResult
from auto_book.models.image import ImageAsset
from auto_book.models.memory import DynamicMemory
from auto_book.models.run_state import ChapterStatus, RunPhase, RunState, TokenUsage
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import is_rate_limit_error

CHECKPOINT_FILENAME = "run_state.json"


def _coerce_phase(value: Any) -> RunPhase:
    if isinstance(value, RunPhase):
        return value
    try:
        return RunPhase(str(value))
    except ValueError:
        return RunPhase.INITIALIZED


def _coerce_model(value: Any, model_type: type, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)


def graph_state_to_run_state(state: dict[str, Any]) -> RunState:
    """Convert a GraphState-like dict into a RunState checkpoint model."""

    book_bible = _coerce_model(state.get("book_bible"), BookBible)
    dynamic_memory = _coerce_model(
        state.get("dynamic_memory"),
        DynamicMemory,
        DynamicMemory(),
    )
    token_usage = _coerce_model(
        state.get("token_usage"),
        TokenUsage,
        TokenUsage(),
    )
    export_result = _coerce_model(state.get("export_result"), ExportResult)

    raw_statuses = state.get("chapter_statuses") or []
    chapter_statuses = [
        item if isinstance(item, ChapterStatus) else ChapterStatus.model_validate(item)
        for item in raw_statuses
    ]

    raw_images = state.get("image_assets") or []
    image_assets = [
        item if isinstance(item, ImageAsset) else ImageAsset.model_validate(item)
        for item in raw_images
    ]

    return RunState(
        run_id=str(state.get("run_id") or "unknown"),
        created_at=state.get("created_at") or datetime.now(),
        updated_at=datetime.now(),
        user_brief=str(state.get("user_brief") or ""),
        genre=str(state.get("genre") or "non-fiction how-to"),
        phase=_coerce_phase(state.get("phase", RunPhase.INITIALIZED)),
        current_chapter=int(state.get("current_chapter") or 0),
        book_bible=book_bible,
        dynamic_memory=dynamic_memory,
        chapter_statuses=chapter_statuses,
        token_usage=token_usage,
        image_assets=image_assets,
        cover_asset=_coerce_model(state.get("cover_asset"), ImageAsset),
        export_result=export_result,
        failure_reason=str(state.get("error") or ""),
        retry_after_seconds=int(state.get("retry_after_seconds") or 0),
        resume_not_before=state.get("resume_not_before"),
        output_directory=str(state.get("output_directory") or "./output"),
    )


def run_state_to_graph_state(run_state: RunState) -> dict[str, Any]:
    """Convert a RunState checkpoint into the graph's initial state shape."""

    phase = run_state.phase
    error = run_state.failure_reason
    current_chapter = run_state.current_chapter

    if phase == RunPhase.RATE_LIMITED or (
        phase == RunPhase.FAILED and is_rate_limit_error(error)
    ):
        current_chapter = _rewind_to_first_unaccepted_chapter(run_state)
        phase = RunPhase.INITIALIZED
        error = ""

    return {
        "run_id": run_state.run_id,
        "created_at": run_state.created_at,
        "updated_at": run_state.updated_at,
        "user_brief": run_state.user_brief,
        "genre": run_state.genre,
        "output_directory": run_state.output_directory,
        "book_bible": run_state.book_bible,
        "current_chapter": current_chapter,
        "chapter_plan": None,
        "chapter_draft": None,
        "review_decision": None,
        "revision_count": 0,
        "regeneration_count": 0,
        "review_passes": 0,
        "dynamic_memory": run_state.dynamic_memory,
        "chapter_statuses": run_state.chapter_statuses,
        "phase": phase,
        "token_usage": run_state.token_usage,
        "image_assets": run_state.image_assets,
        "cover_asset": run_state.cover_asset,
        "export_result": run_state.export_result,
        "retry_after_seconds": run_state.retry_after_seconds,
        "resume_not_before": run_state.resume_not_before,
        "error": error,
    }


def _rewind_to_first_unaccepted_chapter(run_state: RunState) -> int:
    """Return the graph counter value that will prepare the first pending chapter."""

    for status in sorted(run_state.chapter_statuses, key=lambda item: item.chapter_number):
        if status.accepted_draft is None:
            return max(0, status.chapter_number - 1)

    if run_state.book_bible is not None:
        accepted = {
            status.chapter_number
            for status in run_state.chapter_statuses
            if status.accepted_draft is not None
        }
        for chapter in run_state.book_bible.chapter_outline:
            if chapter.chapter_number not in accepted:
                return max(0, chapter.chapter_number - 1)

    return run_state.current_chapter


def save_checkpoint(state: RunState, output_dir: str) -> None:
    """Save the current run state to disk."""

    logger = get_logger()
    path = Path(output_dir) / CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)

    state.updated_at = datetime.now()
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Checkpoint saved to %s", path)


def save_graph_checkpoint(state: dict[str, Any]) -> None:
    """Convert and save a graph state checkpoint."""

    run_state = graph_state_to_run_state(state)
    save_checkpoint(run_state, run_state.output_directory)


def load_checkpoint(output_dir: str) -> RunState | None:
    """Load a run state checkpoint if one exists."""

    logger = get_logger()
    path = Path(output_dir) / CHECKPOINT_FILENAME

    if not path.exists():
        logger.info("No checkpoint found at %s", path)
        return None

    try:
        state = RunState.model_validate_json(path.read_text(encoding="utf-8"))
        logger.info(
            "Checkpoint loaded from %s (phase=%s, chapter=%s)",
            path,
            state.phase.value,
            state.current_chapter,
        )
        return state
    except Exception as exc:
        logger.error("Failed to load checkpoint from %s: %s", path, exc)
        return None
