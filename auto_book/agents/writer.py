"""Writer Agent: writes one chapter at a time."""

import re
import time

from auto_book.agents.llm_client import get_llm
from auto_book.config import settings
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft, ChapterPlan
from auto_book.models.memory import DynamicMemory
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import (
    raise_if_rate_limited,
    record_success,
    wait_for_rate_limit,
)
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
1. Write ONLY the requested chapter.
2. Return ONLY the complete chapter Markdown. Do not return JSON, XML,
   tool/function calls, metadata, commentary, or code fences.
3. Start with a single # chapter heading, use ## for sections.
4. Write between {min_words} and {word_target} words, targeting exactly {word_target}.
   Do NOT exceed {word_target} words. Count your paragraphs: each paragraph is
   ~80-100 words, so {word_target} words means about {paragraph_estimate}
   substantial paragraphs of flowing prose.
5. Write in FLOWING PROSE - full paragraphs with rich detail, anecdotes, and
   examples. DO NOT write bullet-point lists or skeletal outlines. Each
   paragraph should be 3-5 sentences minimum.
6. Use storytelling: open sections with a relatable scenario or question,
   explain concepts through examples, and close with actionable takeaways.
7. Maintain continuity with previous chapters - don't repeat covered material.
8. Match the tone and style guide strictly.
9. Before returning, silently self-review the chapter against this checklist:
   all key topics are covered, examples are concrete, transitions are smooth,
   the promise of the chapter is fulfilled, image anchors follow the rules, and
   the word count is inside the allowed range. Return only the polished final.
{forbidden}
{image_anchor_rules}
"""

WRITER_USER_PROMPT = """Write Chapter {chapter_number}: "{title}".

Chapter goal:
{summary}

Key topics to cover:
{topics}

{continuity_section}

{revision_section}

Write the COMPLETE chapter now as Markdown only. Remember: this draft should be good enough to pass in one review.
"""

REVISION_PREAMBLE = """REVISION INSTRUCTIONS - This is attempt #{revision_number}.
The reviewer identified these specific issues that MUST be fixed:
{feedback}

Your previous draft is shown below. DO NOT start from scratch - instead, REVISE
and EXPAND the existing draft to address each issue above. Keep what works,
fix what doesn't, and add the missing content.

PREVIOUS DRAFT TO REVISE:
---
{previous_body}
---

Now write the REVISED chapter incorporating all fixes:"""

IMAGE_ANCHOR_PATTERN = re.compile(r"^\[IMAGE_ANCHOR:\s*([A-Z0-9_]+)\]\s*$")


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
        min_words=settings.book.min_words_per_chapter,
        paragraph_estimate=paragraph_estimate,
        forbidden=forbidden,
        image_anchor_rules=_build_image_anchor_rules(chapter_plan.chapter_number),
    )

    revision_section = ""
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
            word_target=word_target,
        )

    logger.info(
        "Writer prompt estimate for chapter %s: %s tokens",
        chapter_plan.chapter_number,
        count_tokens(system_msg + user_msg),
    )

    llm = get_llm("writer")

    last_error: Exception | None = None
    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info("Writer attempt %s for chapter %s", attempt, chapter_plan.chapter_number)
            wait_for_rate_limit()
            response = llm.invoke(
                [
                    ("system", system_msg),
                    ("human", user_msg),
                ]
            )
            record_success()
            body = _normalize_markdown_response(
                _extract_message_text(response),
                chapter_plan.title,
            )
            body = _normalize_image_anchors(body)
            if len(body.split()) < max(50, settings.book.min_words_per_chapter // 2):
                raise ValueError("Writer returned an empty or too-short chapter body.")

            result = ChapterDraft(
                chapter_number=chapter_plan.chapter_number,
                title=chapter_plan.title,
                body=body,
                key_takeaways=chapter_plan.key_topics.copy(),
                image_placeholders=[],
                sources_referenced=[],
            )

            errors = validate_chapter_draft(
                result,
                settings.book.min_words_per_chapter,
                word_target,
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
            raise_if_rate_limited(exc)
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


def _build_image_anchor_rules(chapter_number: int) -> str:
    max_images = max(0, settings.images.max_images_per_chapter)
    if not settings.images.enabled or max_images == 0:
        return "\nImage anchor rules:\n- Do not insert any [IMAGE_ANCHOR: ...] markers."

    return f"""
Image anchor rules:
- Insert EXACTLY 2 to {max_images} image anchors per chapter — no fewer than 2, no more than {max_images}.
- Place anchors where a diagram, chart, infographic, or visual would genuinely
  help the reader understand a concept — not just for decoration.
- Each anchor MUST be alone on its own line with this exact format:
  [IMAGE_ANCHOR: CH{chapter_number}_SHORT_DESCRIPTIVE_ID]
- Anchor IDs must use only uppercase letters, numbers, and underscores.
- Make anchor IDs semantic and specific, e.g. CH{chapter_number}_BUDGET_SPLIT_DIAGRAM,
  CH{chapter_number}_WORKFLOW_STEPS, CH{chapter_number}_COMPARISON_TABLE.
- Do not write image prompts, captions, alt text, or image descriptions in the
  chapter. The Image Agent will turn anchors and surrounding text into detailed
  generation prompts later.
- On revisions, preserve existing useful anchors unless they are clearly in the
  wrong place.
"""


def _extract_message_text(response: object) -> str:
    """Extract text from a LangChain chat response."""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _normalize_markdown_response(markdown: str, chapter_title: str) -> str:
    """Clean plain-text LLM output and ensure a stable chapter heading."""

    body = markdown.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()

    expected_heading = f"# {chapter_title}"
    first_line = body.splitlines()[0].strip() if body else ""
    if first_line != expected_heading:
        if first_line.startswith("# "):
            body = "\n".join(body.splitlines()[1:]).strip()
        body = f"{expected_heading}\n\n{body}".strip()

    return body


def _normalize_image_anchors(markdown: str) -> str:
    """Keep only valid standalone anchors and enforce per-chapter limits."""

    max_images = max(0, settings.images.max_images_per_chapter)
    images_enabled = settings.images.enabled and max_images > 0
    lines: list[str] = []
    seen: set[str] = set()
    kept = 0

    for line in markdown.splitlines():
        stripped = line.strip()
        if "[IMAGE_ANCHOR:" not in stripped:
            lines.append(line)
            continue

        if not images_enabled:
            continue

        match = IMAGE_ANCHOR_PATTERN.fullmatch(stripped)
        if not match:
            continue

        anchor_id = match.group(1)
        if anchor_id in seen or kept >= max_images:
            continue

        seen.add(anchor_id)
        kept += 1
        lines.append(f"[IMAGE_ANCHOR: {anchor_id}]")

    return "\n".join(lines).strip()
