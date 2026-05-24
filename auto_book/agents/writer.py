"""Writer Agent: writes one chapter at a time."""

import time

from auto_book.agents.llm_client import get_llm
from auto_book.config import settings
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft, ChapterPlan
from auto_book.models.memory import DynamicMemory
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import wait_for_rate_limit
from auto_book.utils.tokens import count_tokens, truncate_to_budget
from auto_book.utils.validation import validate_chapter_draft

WRITER_SYSTEM_PROMPT = """You are an expert book author writing a {genre} book.

Book context:
- Title: {title}
- Core thesis: {thesis}
- Target audience: {audience}
- Tone: {tone}
- Style guide: {style_guide}

CRITICAL RULES:
1. Write ONLY the requested chapter. Return a ChapterDraft object.
2. Put the FULL chapter text in the "body" field as clean Markdown.
3. Start body with a single # chapter heading, use ## for sections.
4. You MUST write AT LEAST {word_target} words. This is NON-NEGOTIABLE.
   Count your paragraphs: each paragraph is ~80-100 words, so {word_target} words
   means at least {paragraph_estimate} substantial paragraphs of flowing prose.
5. Write in FLOWING PROSE — full paragraphs with rich detail, anecdotes, and
   examples. DO NOT write bullet-point lists or skeletal outlines. Each
   paragraph should be 3-5 sentences minimum.
6. Use storytelling: open sections with a relatable scenario or question,
   explain concepts through examples, and close with actionable takeaways.
7. Maintain continuity with previous chapters — don't repeat covered material.
8. Match the tone and style guide strictly.
{forbidden}
"""

WRITER_USER_PROMPT = """Write Chapter {chapter_number}: "{title}".

Chapter goal:
{summary}

Key topics to cover:
{topics}

{continuity_section}

{revision_section}

{previous_draft_section}

Write the COMPLETE chapter now. Remember: minimum {word_target} words of flowing prose.
"""

REVISION_PREAMBLE = """REVISION INSTRUCTIONS — This is attempt #{revision_number}.
The reviewer identified these specific issues that MUST be fixed:
{feedback}

Your previous draft is shown below. DO NOT start from scratch — instead, REVISE
and EXPAND the existing draft to address each issue above. Keep what works,
fix what doesn't, and add the missing content.

PREVIOUS DRAFT TO REVISE:
---
{previous_body}
---

Now write the REVISED chapter incorporating all fixes:"""


def run_writer(
    chapter_plan: ChapterPlan,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
    revision_feedback: list[str] | None = None,
    previous_draft: ChapterDraft | None = None,
    revision_number: int = 0,
) -> ChapterDraft:
    """Run the live Writer Agent for one chapter."""

    logger = get_logger()
    forbidden = _build_forbidden_section(book_bible)
    continuity = _build_continuity_notes(dynamic_memory)

    word_target = chapter_plan.word_count_target
    paragraph_estimate = max(5, word_target // 90)

    system_msg = WRITER_SYSTEM_PROMPT.format(
        genre=book_bible.genre,
        title=book_bible.working_title,
        thesis=book_bible.core_thesis,
        audience=book_bible.target_audience,
        tone=book_bible.tone,
        style_guide=book_bible.style_guide,
        word_target=word_target,
        paragraph_estimate=paragraph_estimate,
        forbidden=forbidden,
    )

    # Build revision section — now includes the PREVIOUS DRAFT
    revision_section = ""
    previous_draft_section = ""
    if revision_feedback and previous_draft:
        revision_section = REVISION_PREAMBLE.format(
            revision_number=revision_number,
            feedback="\n".join(f"- {item}" for item in revision_feedback),
            previous_body=truncate_to_budget(
                previous_draft.body,
                settings.context_budget.total_max // 3,
            ),
        )
    elif revision_feedback:
        revision_section = _build_revision_section(revision_feedback)

    user_msg = WRITER_USER_PROMPT.format(
        chapter_number=chapter_plan.chapter_number,
        title=chapter_plan.title,
        summary=chapter_plan.summary,
        topics="\n".join(f"- {topic}" for topic in chapter_plan.key_topics),
        continuity_section=continuity,
        revision_section=revision_section,
        previous_draft_section=previous_draft_section,
        word_target=word_target,
    )

    prompt_tokens = count_tokens(system_msg + user_msg)
    if prompt_tokens > settings.context_budget.total_max:
        logger.warning(
            "Writer prompt estimate %s exceeds budget %s; truncating continuity.",
            prompt_tokens,
            settings.context_budget.total_max,
        )
        continuity = truncate_to_budget(
            continuity,
            settings.context_budget.dynamic_memory,
        )
        user_msg = WRITER_USER_PROMPT.format(
            chapter_number=chapter_plan.chapter_number,
            title=chapter_plan.title,
            summary=chapter_plan.summary,
            topics="\n".join(f"- {topic}" for topic in chapter_plan.key_topics),
            continuity_section=continuity,
            revision_section=revision_section,
            previous_draft_section=previous_draft_section,
            word_target=word_target,
        )

    logger.info(
        "Writer prompt estimate for chapter %s: %s tokens",
        chapter_plan.chapter_number,
        count_tokens(system_msg + user_msg),
    )

    llm = get_llm("writer")
    structured_llm = llm.with_structured_output(ChapterDraft)

    last_error: Exception | None = None
    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info("Writer attempt %s for chapter %s", attempt, chapter_plan.chapter_number)
            wait_for_rate_limit()
            result = structured_llm.invoke(
                [
                    ("system", system_msg),
                    ("human", user_msg),
                ]
            )
            if not isinstance(result, ChapterDraft):
                result = ChapterDraft.model_validate(result)

            result.chapter_number = chapter_plan.chapter_number
            if not result.title.strip():
                result.title = chapter_plan.title

            errors = validate_chapter_draft(
                result,
                settings.book.min_words_per_chapter,
                settings.book.max_words_per_chapter,
            )
            if errors and attempt < settings.retry.max_validation_retries:
                raise ValueError("; ".join(errors))
            if errors:
                logger.warning(
                    "Accepting chapter %s with validation warnings: %s",
                    chapter_plan.chapter_number,
                    errors,
                )

            logger.info(
                "Chapter %s drafted: '%s' (%s words)",
                result.chapter_number,
                result.title,
                result.word_count,
            )
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Writer attempt %s for chapter %s failed: %s",
                attempt,
                chapter_plan.chapter_number,
                exc,
            )
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)

    raise ValueError(
        f"Failed to write chapter {chapter_plan.chapter_number} after "
        f"{settings.retry.max_validation_retries} attempts. "
        f"Original error: {last_error}"
    )


def _build_forbidden_section(book_bible: BookBible) -> str:
    if not book_bible.forbidden_topics:
        return ""
    return "\nForbidden topics or claims:\n" + "\n".join(
        f"- {topic}" for topic in book_bible.forbidden_topics
    )


def _build_continuity_notes(memory: DynamicMemory) -> str:
    if not memory.chapters:
        return "Continuity notes:\nThis is the first chapter."

    lines = ["Continuity notes from previous accepted chapters:"]
    for chapter in memory.chapters:
        lines.append(f"- Chapter {chapter.chapter_number}, {chapter.title}: {chapter.summary}")
        if chapter.open_loops:
            lines.append(f"  Open loops: {', '.join(chapter.open_loops)}")

    if memory.repetition_warnings:
        lines.append("\nAvoid repeating:")
        lines.extend(f"- {warning}" for warning in memory.repetition_warnings)

    return truncate_to_budget("\n".join(lines), settings.context_budget.dynamic_memory)


def _build_revision_section(feedback: list[str] | None) -> str:
    if not feedback:
        return ""
    return "Revision feedback to address:\n" + "\n".join(
        f"- {item}" for item in feedback
    )
