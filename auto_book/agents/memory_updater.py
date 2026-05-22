"""Phase 1 stub Dynamic Memory updater."""

from auto_book.models.chapter import ChapterDraft
from auto_book.models.memory import ChapterMemoryEntry, DynamicMemory
from auto_book.utils.logger import get_logger
from auto_book.utils.tokens import count_tokens


def run_memory_update(
    draft: ChapterDraft,
    current_memory: DynamicMemory,
) -> DynamicMemory:
    """Append a deterministic memory entry for an accepted chapter."""

    logger = get_logger()
    updated = current_memory.model_copy(deep=True)
    entry = ChapterMemoryEntry(
        chapter_number=draft.chapter_number,
        title=draft.title,
        summary=(
            f"Accepted Phase 1 stub chapter {draft.chapter_number}: "
            f"{draft.title}."
        ),
        key_claims=[
            "The Phase 1 graph can carry valid chapter state.",
        ],
        characters_or_concepts=["Book Bible", "Dynamic Memory", "ChapterDraft"],
    )
    updated.chapters.append(entry)
    updated.total_word_count += draft.word_count
    updated.repetition_warnings.append(
        f"Chapter {draft.chapter_number} already covered Phase 1 stub flow."
    )
    updated.token_count_estimate = count_tokens(updated.model_dump_json())
    logger.info("Dynamic Memory now has %s chapter entries", len(updated.chapters))
    return updated
