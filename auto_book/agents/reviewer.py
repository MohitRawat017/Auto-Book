"""Reviewer Agent: evaluates chapter quality."""

import json
import time
from json import JSONDecodeError
from typing import Any

from auto_book.agents.llm_client import get_llm
from auto_book.config import settings
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.memory import DynamicMemory
from auto_book.models.review import ReviewDecision
from auto_book.utils.llm import extract_message_text, load_json_object, strip_code_fence
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import (
    raise_if_rate_limited,
    record_success,
    wait_for_rate_limit,
)
from auto_book.utils.tokens import count_tokens, truncate_to_budget
from auto_book.utils.validation import validate_review_decision

REVIEWER_SYSTEM_PROMPT = """You are a strict but constructive book editor.

Evaluate a chapter draft against the Book Bible and previous chapter summaries.

SCORING CALIBRATION - follow these examples precisely:
- Score 9-10 (pass): Excellent. Reads like a polished book chapter. Rich prose,
  strong examples, good flow, meets the {word_target}-word target. Very rare.
- Score 7-8 (pass): Good. Covers the topic well, has decent prose, roughly meets
  the {word_target}-word target (within 20%). Minor suggestions only.
  THIS IS THE EXPECTED RANGE for a competent draft.
- Score 5-6 (revise): Has structural problems: missing key topics, OR more than
  20% below the {word_target}-word target, OR reads like a bullet-point outline.
  Requires specific fixes.
- Score 0-4 (fail): Off-topic, incoherent, or below 50% of the {word_target}-word
  target. Needs full regeneration.

DECISION RULES:
- pass (score 7-10): Chapter is ready. Put minor polish suggestions in
  suggested_edits and return pass. DO NOT return revise for subjective
  improvements like "could use more examples" if the chapter already covers
  the topic adequately.
- revise (score 5-6): Chapter has specific, fixable structural problems.
  You MUST list concrete required_fixes. Only request revisions for
  OBJECTIVE issues, not stylistic preferences.
- fail (score 0-4): Chapter misses the goal entirely. Needs full regeneration.

{review_policy}

Return ONLY valid JSON. Do not return Markdown, code fences, comments, or a
tool/function call. Every list field must be a JSON array of strings.

Required JSON shape:
{{
  "chapter_number": {chapter_number},
  "decision": "<pass|revise|fail>",
  "score": "<integer 0-10>",
  "problems": [],
  "required_fixes": [],
  "suggested_edits": [],
  "continuity_issues": [],
  "research_gaps": [],
  "tone_match": true
}}

{revision_context}

Book context:
- Title: {title}
- Thesis: {thesis}
- Audience: {audience}
- Tone: {tone}
- Style guide: {style_guide}

Chapter goal:
Chapter {chapter_number}: "{chapter_title}"
Expected content: {chapter_summary}
Word target: {word_target}
"""

REVIEWER_USER_PROMPT = """Review this chapter draft:

{chapter_body}

{continuity_context}

Return only the ReviewDecision JSON. Be fair - if the chapter covers its topic
with reasonable prose and approximately meets the word target, pass it.
"""

LIST_FIELDS = (
    "problems",
    "required_fixes",
    "suggested_edits",
    "continuity_issues",
    "research_gaps",
)


def run_reviewer(
    draft: ChapterDraft,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
    revision_count: int = 0,
) -> ReviewDecision:
    """Run the live Reviewer Agent."""

    logger = get_logger()
    outline = next(
        (
            chapter
            for chapter in book_bible.chapter_outline
            if chapter.chapter_number == draft.chapter_number
        ),
        None,
    )
    chapter_title = outline.title if outline else draft.title
    chapter_summary = outline.summary if outline else "No chapter summary available."
    word_target = outline.word_count_target if outline else settings.book.target_words_per_chapter

    revision_context = ""
    if revision_count >= 2:
        revision_context = (
            "IMPORTANT: This chapter has already been revised "
            f"{revision_count} time(s). Be MORE LENIENT in your scoring. "
            "If the chapter covers the core topic and is reasonably well-written, "
            "score it 7+ and pass it. Do NOT ask for further revisions unless "
            "there are critical structural problems. Minor improvements should "
            "go in suggested_edits with a pass decision."
        )
    elif revision_count == 1:
        revision_context = (
            "NOTE: This is a revised draft (revision #1). The writer addressed "
            "previous feedback. If the main issues were fixed, score 7+ and pass. "
            "Only request another revision for remaining serious problems."
        )

    review_policy = _build_review_policy()

    system_msg = REVIEWER_SYSTEM_PROMPT.format(
        title=book_bible.working_title,
        thesis=book_bible.core_thesis,
        audience=book_bible.target_audience,
        tone=book_bible.tone,
        style_guide=book_bible.style_guide,
        chapter_number=draft.chapter_number,
        chapter_title=chapter_title,
        chapter_summary=chapter_summary,
        word_target=word_target,
        revision_context=revision_context,
        review_policy=review_policy,
    )
    user_msg = REVIEWER_USER_PROMPT.format(
        chapter_body=truncate_to_budget(draft.body, settings.context_budget.total_max // 2),
        continuity_context=_build_review_continuity(dynamic_memory),
    )
    logger.info(
        "Reviewer prompt estimate for chapter %s: %s tokens",
        draft.chapter_number,
        count_tokens(system_msg + user_msg),
    )

    llm = get_llm("reviewer")

    last_error: Exception | None = None
    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info("Reviewer attempt %s for chapter %s", attempt, draft.chapter_number)
            wait_for_rate_limit()
            response = llm.invoke(
                [
                    ("system", system_msg),
                    ("human", user_msg),
                ]
            )
            record_success()
            result = _parse_review_decision(
                extract_message_text(response),
                draft.chapter_number,
            )

            errors = validate_review_decision(result)
            if errors:
                raise ValueError("; ".join(errors))

            logger.info(
                "Chapter %s review: %s (score %.1f)",
                result.chapter_number,
                result.decision.value,
                result.score,
            )
            return result
        except Exception as exc:
            raise_if_rate_limited(exc)
            last_error = exc
            logger.warning(
                "Reviewer attempt %s for chapter %s failed: %s",
                attempt,
                draft.chapter_number,
                exc,
            )
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)

    raise ValueError(
        f"Failed to review chapter {draft.chapter_number} after "
        f"{settings.retry.max_validation_retries} attempts. "
        f"Original error: {last_error}"
    )


def _build_review_continuity(memory: DynamicMemory) -> str:
    if not memory.chapters:
        return "Previous chapters: none."

    lines = ["Previous accepted chapters:"]
    for chapter in memory.chapters:
        lines.append(f"- Chapter {chapter.chapter_number}, {chapter.title}: {chapter.summary}")

    if memory.repetition_warnings:
        lines.append("\nKnown repetition risks:")
        lines.extend(f"- {warning}" for warning in memory.repetition_warnings)

    return truncate_to_budget("\n".join(lines), settings.context_budget.dynamic_memory)


def _build_review_policy() -> str:
    if settings.review.max_review_passes > 1:
        return ""

    return (
        "BUDGET-CONSCIOUS REVIEW MODE:\n"
        f"- The reviewer runs at most once per chapter.\n"
        f"- Treat {settings.review.accept_score:.1f}+ as a clean pass.\n"
        f"- Also return pass for any usable chapter scoring {settings.review.soft_accept_score:.1f} or higher.\n"
        "- Put non-critical improvements in suggested_edits instead of requesting revision.\n"
        "- Use revise only for objective, serious issues that make the chapter unsuitable.\n"
        "- Use fail only when the chapter is off-topic, incoherent, or far below the requested scope."
    )


def _parse_review_decision(raw_text: str, chapter_number: int) -> ReviewDecision:
    payload = load_json_object(strip_code_fence(raw_text))
    if isinstance(payload.get("arguments"), dict):
        payload = payload["arguments"]

    payload["chapter_number"] = chapter_number
    for field in LIST_FIELDS:
        payload[field] = _coerce_string_list(payload.get(field))
    payload["tone_match"] = bool(payload.get("tone_match", True))

    return ReviewDecision.model_validate(payload)


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        lines = [line.strip(" -0123456789.") for line in value.splitlines()]
        return [line for line in lines if line]
    return [str(value).strip()] if str(value).strip() else []
