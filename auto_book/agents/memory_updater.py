"""Memory Updater Agent: updates Dynamic Memory after accepted chapters."""

import time

from auto_book.agents.llm_client import get_llm
from auto_book.config import settings
from auto_book.models.chapter import ChapterDraft
from auto_book.models.memory import ChapterMemoryEntry, DynamicMemory
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import (
    raise_if_rate_limited,
    record_success,
    wait_for_rate_limit,
)
from auto_book.utils.tokens import count_tokens, truncate_to_budget

MEMORY_SYSTEM_PROMPT = """You are a meticulous editorial assistant.

Summarize an accepted book chapter into a structured memory entry that future
chapters can use for continuity.

Rules:
- Summary should be 3-5 sentences.
- Capture important claims, definitions, concepts, examples, and open loops.
- Do not add information that is not in the chapter.
- Be concise and factual.
"""

MEMORY_USER_PROMPT = """Create a ChapterMemoryEntry for this accepted chapter.

Chapter {chapter_number}: "{title}"

{body}
"""


def run_memory_update(
    draft: ChapterDraft,
    current_memory: DynamicMemory,
) -> DynamicMemory:
    """Run the live Memory Updater Agent."""

    logger = get_logger()
    llm = get_llm("memory")
    structured_llm = llm.with_structured_output(ChapterMemoryEntry)

    body = truncate_to_budget(draft.body, settings.context_budget.total_max // 2)
    user_msg = MEMORY_USER_PROMPT.format(
        chapter_number=draft.chapter_number,
        title=draft.title,
        body=body,
    )
    logger.info(
        "Memory prompt estimate for chapter %s: %s tokens",
        draft.chapter_number,
        count_tokens(MEMORY_SYSTEM_PROMPT + user_msg),
    )

    last_error: Exception | None = None
    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info("Memory update attempt %s for chapter %s", attempt, draft.chapter_number)
            wait_for_rate_limit()
            entry = structured_llm.invoke(
                [
                    ("system", MEMORY_SYSTEM_PROMPT),
                    ("human", user_msg),
                ]
            )
            record_success()
            if not isinstance(entry, ChapterMemoryEntry):
                entry = ChapterMemoryEntry.model_validate(entry)

            entry.chapter_number = draft.chapter_number
            entry.title = draft.title
            updated = _append_entry(current_memory, entry, draft.word_count)
            logger.info(
                "Dynamic Memory updated: %s chapters, approx %s tokens",
                len(updated.chapters),
                updated.token_count_estimate,
            )
            return updated
        except Exception as exc:
            raise_if_rate_limited(exc)
            last_error = exc
            logger.warning(
                "Memory update attempt %s for chapter %s failed: %s",
                attempt,
                draft.chapter_number,
                exc,
            )
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)

    logger.warning(
        "Using fallback memory entry for chapter %s after error: %s",
        draft.chapter_number,
        last_error,
    )
    fallback = ChapterMemoryEntry(
        chapter_number=draft.chapter_number,
        title=draft.title,
        summary=(
            f"Chapter {draft.chapter_number}, {draft.title}, was accepted. "
            "Automatic summary was unavailable."
        ),
    )
    return _append_entry(current_memory, fallback, draft.word_count)


def _append_entry(
    current_memory: DynamicMemory,
    entry: ChapterMemoryEntry,
    word_count: int,
) -> DynamicMemory:
    updated = current_memory.model_copy(deep=True)
    updated.chapters.append(entry)
    updated.total_word_count += word_count
    for claim in entry.key_claims:
        warning = f"Chapter {entry.chapter_number} already covered: {claim}"
        if warning not in updated.repetition_warnings:
            updated.repetition_warnings.append(warning)
    updated.token_count_estimate = count_tokens(updated.model_dump_json())
    return _compress_memory(updated)


def _compress_memory(memory: DynamicMemory) -> DynamicMemory:
    budget = settings.context_budget.dynamic_memory
    if memory.token_count_estimate <= budget * 2 or len(memory.chapters) <= 3:
        return memory

    compressed = memory.model_copy(deep=True)
    for entry in compressed.chapters[:-3]:
        entry.key_claims = entry.key_claims[:2]
        entry.definitions_introduced = []
        entry.characters_or_concepts = entry.characters_or_concepts[:2]
        entry.open_loops = entry.open_loops[:2]
        entry.research_facts_used = []

    compressed.token_count_estimate = count_tokens(compressed.model_dump_json())
    get_logger().info("Dynamic Memory compressed to approx %s tokens", compressed.token_count_estimate)
    return compressed
