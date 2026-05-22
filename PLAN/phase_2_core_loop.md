# Phase 2 — Core Writing Loop

## Goal

Replace all stub agents with live LLM-powered agents using LangChain + Groq. The system should be able to take a user brief, plan a book, write chapters one-by-one with review and retry, maintain dynamic memory, and produce a final assembled Markdown book.

**Prerequisite**: Phase 1 is fully complete and the stub orchestrator runs end-to-end.

**Success Criteria**: Given a topic like "A beginner's guide to personal finance", the system autonomously produces a coherent ~30-page Markdown book with 8-12 chapters, consistent tone, and no major repetition.

---

## 2.1 — LLM Client Setup

### File: `auto_book/agents/llm_client.py` (NEW)

Create a centralized LLM client module that all agents import from. This avoids duplicating Groq initialization logic.

```python
"""
Centralized LLM client factory.

Provides pre-configured ChatGroq instances for different agent roles.
Each role uses the model and temperature specified in config.yaml.
"""

from langchain_groq import ChatGroq
from auto_book.config import secrets, settings


def get_llm(role: str) -> ChatGroq:
    """
    Get a configured ChatGroq instance for a specific agent role.

    Args:
        role: One of "writer", "planner", "reviewer", "memory"

    Returns:
        A ChatGroq instance with the appropriate model and temperature.
    """
    model_map = {
        "writer": settings.llm.writer_model,
        "planner": settings.llm.planner_model,
        "reviewer": settings.llm.reviewer_model,
        "memory": settings.llm.memory_model,
    }
    temp_map = {
        "writer": settings.llm.temperature_creative,
        "planner": settings.llm.temperature_creative,
        "reviewer": settings.llm.temperature_analytical,
        "memory": settings.llm.temperature_analytical,
    }

    model = model_map.get(role, settings.llm.writer_model)
    temperature = temp_map.get(role, 0.5)

    return ChatGroq(
        api_key=secrets.groq_api_key,
        model=model,
        temperature=temperature,
    )
```

---

## 2.2 — Planner Agent

### File: `auto_book/agents/planner.py`

The Planner Agent takes the user brief + genre and produces a complete `BookBible`.

**Implementation details:**

1. Build a system prompt that instructs the LLM to act as a book planning expert.
2. Include the user brief and genre in the user message.
3. Use LangChain's `.with_structured_output(BookBible)` to force Pydantic-compliant JSON output.
4. Validate the output: ensure `chapter_outline` has the expected number of chapters, all fields are non-empty, and chapter numbers are sequential.
5. If validation fails, retry up to `config.retry.max_validation_retries` times.

```python
"""
Planner Agent — creates the Book Bible from a user brief.

Input: user_brief (str), genre (str)
Output: BookBible (Pydantic model)
"""

import time
from auto_book.agents.llm_client import get_llm
from auto_book.models.book_bible import BookBible
from auto_book.config import settings
from auto_book.utils.logger import get_logger
from auto_book.utils.tokens import count_tokens

PLANNER_SYSTEM_PROMPT = """You are an expert book planner and publishing strategist.

Your job is to create a comprehensive Book Bible — a planning document that will guide the writing of an entire book.

RULES:
- The book should have {chapter_count} chapters.
- Each chapter should target approximately {words_per_chapter} words.
- The total book should be approximately {total_pages} pages.
- Create a clear, compelling thesis that ties the whole book together.
- Define a specific target reader — not just "everyone".
- The tone and style should be consistent and appropriate for the genre.
- Each chapter should build on previous chapters logically.
- Include at least 3 reader pain points that the book addresses.
- The chapter outline must have sequential chapter numbers starting from 1.
- Each chapter must have a clear, distinct purpose — no redundant chapters.
- Include forbidden topics to prevent the writer from going off-track.
- The glossary should define key terms that must be used consistently.

GENRE: {genre}
"""

PLANNER_USER_PROMPT = """Create a complete Book Bible for the following book idea:

{brief}

Return a comprehensive plan covering: title, thesis, audience, tone, style guide, 
chapter outline with summaries and key topics, required themes, forbidden topics, 
and a glossary of key terms.
"""


def run_planner(user_brief: str, genre: str) -> BookBible:
    """
    Run the Planner Agent to create a Book Bible.

    Args:
        user_brief: The user's book idea/topic description.
        genre: The genre/type of book.

    Returns:
        A validated BookBible instance.

    Raises:
        ValueError: If the planner fails after all retries.
    """
    logger = get_logger()
    llm = get_llm("planner")
    structured_llm = llm.with_structured_output(BookBible)

    system_msg = PLANNER_SYSTEM_PROMPT.format(
        chapter_count=settings.book.default_chapter_count,
        words_per_chapter=settings.book.target_words_per_chapter,
        total_pages=settings.book.target_total_pages,
        genre=genre,
    )
    user_msg = PLANNER_USER_PROMPT.format(brief=user_brief)

    # Log token usage estimate
    prompt_tokens = count_tokens(system_msg + user_msg)
    logger.info(f"Planner prompt: ~{prompt_tokens} tokens")

    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info(f"Planner attempt {attempt}")
            result = structured_llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

            # Validate beyond schema
            errors = _validate_book_bible(result)
            if errors:
                logger.warning(f"Planner output validation errors: {errors}")
                if attempt < settings.retry.max_validation_retries:
                    time.sleep(settings.retry.retry_delay_seconds)
                    continue
                else:
                    logger.error("Planner failed validation after all retries")
                    raise ValueError(f"Planner validation failed: {errors}")

            logger.info(f"Book Bible created: '{result.working_title}' with {len(result.chapter_outline)} chapters")
            return result

        except Exception as e:
            logger.error(f"Planner attempt {attempt} failed: {e}")
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)
            else:
                raise ValueError(f"Planner failed after {attempt} attempts: {e}")

    raise ValueError("Planner failed — should not reach here")


def _validate_book_bible(bible: BookBible) -> list[str]:
    """Semantic validation of a BookBible beyond Pydantic schema."""
    errors = []
    if not bible.working_title.strip():
        errors.append("Working title is empty")
    if not bible.core_thesis.strip():
        errors.append("Core thesis is empty")
    if not bible.target_audience.strip():
        errors.append("Target audience is empty")
    if len(bible.chapter_outline) == 0:
        errors.append("Chapter outline is empty")
    if len(bible.chapter_outline) < 3:
        errors.append(f"Only {len(bible.chapter_outline)} chapters — need at least 3")

    # Check sequential chapter numbers
    for i, ch in enumerate(bible.chapter_outline):
        if ch.chapter_number != i + 1:
            errors.append(f"Chapter numbers not sequential: expected {i+1}, got {ch.chapter_number}")
            break

    # Check for duplicate titles
    titles = [ch.title for ch in bible.chapter_outline]
    if len(titles) != len(set(titles)):
        errors.append("Duplicate chapter titles found")

    return errors
```

### Orchestrator integration

Replace the stub `plan_book` node in `orchestrator/graph.py`:

```python
def plan_book(state: GraphState) -> dict:
    """Node 2: Planner Agent creates Book Bible."""
    logger = get_logger()
    try:
        from auto_book.agents.planner import run_planner
        bible = run_planner(state["user_brief"], state["genre"])

        # Initialize chapter statuses
        chapter_statuses = [
            ChapterStatus(chapter_number=ch.chapter_number)
            for ch in bible.chapter_outline
        ]

        return {
            "book_bible": bible,
            "chapter_statuses": chapter_statuses,
            "phase": RunPhase.PLANNING,
        }
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {"phase": RunPhase.FAILED, "error": str(e)}
```

---

## 2.3 — Writer Agent

### File: `auto_book/agents/writer.py`

The Writer Agent writes one chapter at a time using the Book Bible, Dynamic Memory, and the chapter plan.

**Critical implementation details:**

1. **Context window budget enforcement**: Before calling the LLM, compute the token count of each context component (Book Bible excerpt, Dynamic Memory, chapter plan). If any component exceeds its budget from `config.context_budget`, truncate it using `utils/tokens.py`.
2. **Revision mode**: When `revision_feedback` is provided (from a failed review), include it in the prompt and instruct the LLM to fix the specific issues.
3. **Continuity**: Include summaries of previous chapters from Dynamic Memory so the Writer knows what has already been covered.

```python
"""
Writer Agent — writes one chapter at a time.

Input: ChapterPlan, BookBible (excerpt), DynamicMemory, optional revision feedback
Output: ChapterDraft (Pydantic model)
"""

import time
from auto_book.agents.llm_client import get_llm
from auto_book.models.chapter import ChapterDraft, ChapterPlan
from auto_book.models.book_bible import BookBible
from auto_book.models.memory import DynamicMemory
from auto_book.config import settings
from auto_book.utils.logger import get_logger
from auto_book.utils.tokens import count_tokens, truncate_to_budget
from auto_book.utils.validation import validate_chapter_draft


WRITER_SYSTEM_PROMPT = """You are an expert book author writing a {genre} book.

BOOK CONTEXT:
Title: {title}
Core Thesis: {thesis}
Target Audience: {audience}
Tone: {tone}
Style Guide: {style_guide}

RULES:
- Write ONLY the chapter specified below. Do not write other chapters.
- Target approximately {word_target} words for this chapter.
- Write in clean Markdown format.
- Use section headings (##) within the chapter where appropriate.
- Maintain consistency with previous chapters (see CONTINUITY NOTES below).
- Do NOT repeat points already covered in previous chapters.
- Follow the tone and style guide strictly.
- If this is a revision, focus on fixing the specific issues listed.
{forbidden}
"""

WRITER_CHAPTER_PROMPT = """Write Chapter {chapter_number}: "{title}"

CHAPTER GOAL:
{summary}

KEY TOPICS TO COVER:
{topics}

{continuity_section}

{revision_section}

Write the complete chapter now. Start with the chapter title as a Markdown heading.
"""


def run_writer(
    chapter_plan: ChapterPlan,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
    revision_feedback: list[str] | None = None,
) -> ChapterDraft:
    """
    Run the Writer Agent to draft a single chapter.

    Args:
        chapter_plan: The plan for this specific chapter.
        book_bible: The Book Bible (will be excerpted to fit budget).
        dynamic_memory: Current dynamic memory state.
        revision_feedback: Optional list of fixes from a failed review.

    Returns:
        A validated ChapterDraft instance.

    Raises:
        ValueError: If writing fails after all retries.
    """
    logger = get_logger()
    llm = get_llm("writer")
    structured_llm = llm.with_structured_output(ChapterDraft)

    # Build context components with token budget enforcement
    bible_excerpt = _build_bible_excerpt(book_bible)
    continuity = _build_continuity_notes(dynamic_memory)
    revision_section = _build_revision_section(revision_feedback)
    forbidden = ""
    if book_bible.forbidden_topics:
        forbidden = "\nFORBIDDEN TOPICS (do not mention): " + ", ".join(book_bible.forbidden_topics)

    system_msg = WRITER_SYSTEM_PROMPT.format(
        genre=book_bible.genre,
        title=book_bible.working_title,
        thesis=book_bible.core_thesis,
        audience=book_bible.target_audience,
        tone=book_bible.tone,
        style_guide=book_bible.style_guide,
        word_target=chapter_plan.word_count_target,
        forbidden=forbidden,
    )
    user_msg = WRITER_CHAPTER_PROMPT.format(
        chapter_number=chapter_plan.chapter_number,
        title=chapter_plan.title,
        summary=chapter_plan.summary,
        topics="\n".join(f"- {t}" for t in chapter_plan.key_topics),
        continuity_section=continuity,
        revision_section=revision_section,
    )

    # Enforce total token budget
    total_prompt_tokens = count_tokens(system_msg + user_msg)
    budget = settings.context_budget.total_max
    if total_prompt_tokens > budget:
        logger.warning(
            f"Writer prompt ({total_prompt_tokens} tokens) exceeds budget ({budget}). "
            f"Truncating dynamic memory."
        )
        # Truncate memory first (it's the most compressible component)
        continuity = truncate_to_budget(continuity, settings.context_budget.dynamic_memory)
        user_msg = WRITER_CHAPTER_PROMPT.format(
            chapter_number=chapter_plan.chapter_number,
            title=chapter_plan.title,
            summary=chapter_plan.summary,
            topics="\n".join(f"- {t}" for t in chapter_plan.key_topics),
            continuity_section=continuity,
            revision_section=revision_section,
        )

    logger.info(f"Writer prompt: ~{count_tokens(system_msg + user_msg)} tokens")

    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            logger.info(f"Writer attempt {attempt} for chapter {chapter_plan.chapter_number}")
            result = structured_llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

            # Ensure chapter_number matches
            result.chapter_number = chapter_plan.chapter_number

            # Semantic validation
            errors = validate_chapter_draft(
                result,
                min_words=settings.book.min_words_per_chapter,
                max_words=settings.book.max_words_per_chapter,
            )
            if errors:
                logger.warning(f"Writer draft validation errors: {errors}")
                if attempt < settings.retry.max_validation_retries:
                    time.sleep(settings.retry.retry_delay_seconds)
                    continue
                else:
                    # Accept with warnings if it's just word count issues
                    logger.warning("Accepting draft with validation warnings after max retries")

            logger.info(
                f"Chapter {result.chapter_number} draft: "
                f"'{result.title}' — {result.word_count} words"
            )
            return result

        except Exception as e:
            logger.error(f"Writer attempt {attempt} failed: {e}")
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)
            else:
                raise ValueError(f"Writer failed after {attempt} attempts: {e}")

    raise ValueError("Writer failed — should not reach here")


def _build_bible_excerpt(bible: BookBible) -> str:
    """Build a token-budgeted excerpt of the Book Bible for the Writer."""
    excerpt = (
        f"Title: {bible.working_title}\n"
        f"Thesis: {bible.core_thesis}\n"
        f"Audience: {bible.target_audience}\n"
        f"Tone: {bible.tone}\n"
        f"Style: {bible.style_guide}\n"
    )
    return truncate_to_budget(excerpt, settings.context_budget.book_bible)


def _build_continuity_notes(memory: DynamicMemory) -> str:
    """Build continuity notes from Dynamic Memory."""
    if not memory.chapters:
        return "CONTINUITY NOTES:\nThis is the first chapter. No previous content."

    lines = ["CONTINUITY NOTES (what previous chapters covered):"]
    for ch in memory.chapters:
        lines.append(f"\n- Chapter {ch.chapter_number} '{ch.title}': {ch.summary}")
        if ch.open_loops:
            lines.append(f"  Open threads: {', '.join(ch.open_loops)}")

    if memory.repetition_warnings:
        lines.append("\nDO NOT REPEAT:")
        for warning in memory.repetition_warnings:
            lines.append(f"- {warning}")

    result = "\n".join(lines)
    return truncate_to_budget(result, settings.context_budget.dynamic_memory)


def _build_revision_section(feedback: list[str] | None) -> str:
    """Build revision instructions from reviewer feedback."""
    if not feedback:
        return ""
    lines = ["REVISION REQUIRED — Fix these specific issues:"]
    for fix in feedback:
        lines.append(f"- {fix}")
    return "\n".join(lines)
```

### Orchestrator integration

Replace the stub `write_chapter` and `prepare_chapter` nodes:

```python
def prepare_chapter(state: GraphState) -> dict:
    """Node 3: Extract the current chapter's plan from the Book Bible."""
    logger = get_logger()
    bible = state["book_bible"]
    current = state.get("current_chapter", 0) + 1
    
    # Find the chapter outline entry
    outline = bible.chapter_outline[current - 1]
    
    chapter_plan = ChapterPlan(
        chapter_number=outline.chapter_number,
        title=outline.title,
        summary=outline.summary,
        key_topics=outline.key_topics,
        word_count_target=outline.word_count_target,
    )
    
    logger.info(f"Preparing chapter {current}: '{chapter_plan.title}'")
    return {
        "current_chapter": current,
        "chapter_plan": chapter_plan,
        "revision_count": 0,
        "regeneration_count": 0,
        "phase": RunPhase.WRITING,
    }


def write_chapter(state: GraphState) -> dict:
    """Node 4: Writer Agent drafts a chapter."""
    logger = get_logger()
    try:
        from auto_book.agents.writer import run_writer
        
        # Determine if this is a revision
        review = state.get("review_decision")
        revision_feedback = None
        revision_count = state.get("revision_count", 0)
        regeneration_count = state.get("regeneration_count", 0)
        
        if review and review.decision.value == "revise":
            revision_feedback = review.required_fixes
            revision_count += 1
        elif review and review.decision.value == "fail":
            regeneration_count += 1
        
        draft = run_writer(
            chapter_plan=state["chapter_plan"],
            book_bible=state["book_bible"],
            dynamic_memory=state.get("dynamic_memory", DynamicMemory()),
            revision_feedback=revision_feedback,
        )
        
        return {
            "chapter_draft": draft,
            "revision_count": revision_count,
            "regeneration_count": regeneration_count,
            "phase": RunPhase.WRITING,
        }
    except Exception as e:
        logger.error(f"Writing failed: {e}")
        return {"error": str(e), "phase": RunPhase.FAILED}
```

---

## 2.4 — Reviewer Agent

### File: `auto_book/agents/reviewer.py`

The Reviewer evaluates a chapter draft against the Book Bible and Dynamic Memory.

```python
"""
Reviewer Agent — evaluates chapter quality against the Book Bible.

Input: ChapterDraft, BookBible, DynamicMemory
Output: ReviewDecision (Pydantic model)
"""

import time
from auto_book.agents.llm_client import get_llm
from auto_book.models.chapter import ChapterDraft
from auto_book.models.book_bible import BookBible
from auto_book.models.memory import DynamicMemory
from auto_book.models.review import ReviewDecision
from auto_book.config import settings
from auto_book.utils.logger import get_logger
from auto_book.utils.tokens import count_tokens, truncate_to_budget
from auto_book.utils.validation import validate_review_decision


REVIEWER_SYSTEM_PROMPT = """You are a strict but constructive book editor and reviewer.

Your job is to evaluate a chapter draft against the Book Bible and previous chapter summaries.

EVALUATION CRITERIA:
1. Does the chapter achieve its stated goal/summary from the outline?
2. Is the tone consistent with the Book Bible style guide?
3. Does it avoid repeating content from previous chapters?
4. Are claims supported or reasonable?
5. Is the structure clear with proper sections?
6. Is the word count appropriate (target: {word_target})?
7. Does it maintain continuity with previous chapters?
8. Is it engaging for the target audience: {audience}?

SCORING:
- 8-10: Pass — chapter is ready. Minor suggestions only.
- 5-7: Revise — chapter has fixable issues. Provide specific required_fixes.
- 0-4: Fail — chapter misses the goal significantly. Needs full regeneration.

BOOK BIBLE CONTEXT:
Title: {title}
Thesis: {thesis}
Tone: {tone}
Style Guide: {style_guide}

CHAPTER GOAL:
Chapter {chapter_number}: "{chapter_title}"
Expected content: {chapter_summary}
"""

REVIEWER_USER_PROMPT = """Review the following chapter draft:

{chapter_body}

{continuity_context}

Evaluate this chapter against the criteria and provide your structured review decision.
"""


def run_reviewer(
    draft: ChapterDraft,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
) -> ReviewDecision:
    """
    Run the Reviewer Agent to evaluate a chapter draft.

    Args:
        draft: The chapter draft to review.
        book_bible: The Book Bible for consistency checks.
        dynamic_memory: Previous chapter summaries for repetition/continuity checks.

    Returns:
        A validated ReviewDecision instance.
    """
    logger = get_logger()
    llm = get_llm("reviewer")
    structured_llm = llm.with_structured_output(ReviewDecision)

    # Find the chapter outline entry
    outline_entry = None
    for ch in book_bible.chapter_outline:
        if ch.chapter_number == draft.chapter_number:
            outline_entry = ch
            break

    chapter_summary = outline_entry.summary if outline_entry else "No specific summary"
    chapter_title = outline_entry.title if outline_entry else draft.title

    system_msg = REVIEWER_SYSTEM_PROMPT.format(
        word_target=settings.book.target_words_per_chapter,
        audience=book_bible.target_audience,
        title=book_bible.working_title,
        thesis=book_bible.core_thesis,
        tone=book_bible.tone,
        style_guide=book_bible.style_guide,
        chapter_number=draft.chapter_number,
        chapter_title=chapter_title,
        chapter_summary=chapter_summary,
    )

    # Build continuity context from memory
    continuity = _build_review_continuity(dynamic_memory)

    user_msg = REVIEWER_USER_PROMPT.format(
        chapter_body=draft.body[:6000],  # Cap chapter text to avoid overflow
        continuity_context=continuity,
    )

    logger.info(f"Reviewing chapter {draft.chapter_number}: ~{count_tokens(system_msg + user_msg)} tokens")

    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            result = structured_llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

            result.chapter_number = draft.chapter_number

            # Validate the review
            errors = validate_review_decision(result)
            if errors:
                logger.warning(f"Review validation errors: {errors}")
                if attempt < settings.retry.max_validation_retries:
                    time.sleep(settings.retry.retry_delay_seconds)
                    continue

            logger.info(
                f"Chapter {result.chapter_number} review: "
                f"{result.decision.value} (score: {result.score})"
            )
            return result

        except Exception as e:
            logger.error(f"Reviewer attempt {attempt} failed: {e}")
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)
            else:
                raise ValueError(f"Reviewer failed after {attempt} attempts: {e}")

    raise ValueError("Reviewer failed — should not reach here")


def _build_review_continuity(memory: DynamicMemory) -> str:
    """Build previous chapter summaries for the reviewer."""
    if not memory.chapters:
        return "PREVIOUS CHAPTERS: None (this is the first chapter)"

    lines = ["PREVIOUS CHAPTERS (check for repetition and continuity):"]
    for ch in memory.chapters:
        lines.append(f"- Ch {ch.chapter_number} '{ch.title}': {ch.summary}")
    
    if memory.repetition_warnings:
        lines.append("\nKNOWN REPETITION RISKS:")
        for w in memory.repetition_warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)
```

---

## 2.5 — Memory Updater Agent

### File: `auto_book/agents/memory_updater.py`

After a chapter is accepted, this agent summarizes it and updates Dynamic Memory.

```python
"""
Memory Updater — updates Dynamic Memory after a chapter is accepted.

Input: ChapterDraft (accepted), existing DynamicMemory
Output: Updated DynamicMemory
"""

import time
from auto_book.agents.llm_client import get_llm
from auto_book.models.chapter import ChapterDraft
from auto_book.models.memory import DynamicMemory, ChapterMemoryEntry
from auto_book.config import settings
from auto_book.utils.logger import get_logger
from auto_book.utils.tokens import count_tokens


MEMORY_SYSTEM_PROMPT = """You are a meticulous editorial assistant.

Your job is to summarize an accepted book chapter into a structured memory entry.
This entry will be used by the writer of FUTURE chapters to maintain continuity.

RULES:
- The summary should be 3-5 sentences capturing the key points.
- List any important claims, definitions, or concepts introduced.
- Note any characters, examples, or case studies introduced by name.
- Identify open loops: questions raised but not answered, promises made for later chapters.
- List specific facts that were used so the writer doesn't repeat them.
- Be precise and factual — do not add information not in the chapter.
"""

MEMORY_USER_PROMPT = """Summarize the following accepted chapter into a structured memory entry:

Chapter {chapter_number}: "{title}"

{body}
"""


def run_memory_update(
    draft: ChapterDraft,
    current_memory: DynamicMemory,
) -> DynamicMemory:
    """
    Update Dynamic Memory with a summary of the accepted chapter.

    Args:
        draft: The accepted chapter draft.
        current_memory: The current state of Dynamic Memory.

    Returns:
        Updated DynamicMemory with the new chapter entry.
    """
    logger = get_logger()
    llm = get_llm("memory")
    structured_llm = llm.with_structured_output(ChapterMemoryEntry)

    system_msg = MEMORY_SYSTEM_PROMPT
    user_msg = MEMORY_USER_PROMPT.format(
        chapter_number=draft.chapter_number,
        title=draft.title,
        body=draft.body[:5000],  # Cap to avoid token overflow
    )

    logger.info(f"Updating memory for chapter {draft.chapter_number}")

    for attempt in range(1, settings.retry.max_validation_retries + 1):
        try:
            entry = structured_llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

            entry.chapter_number = draft.chapter_number
            entry.title = draft.title

            # Update the memory object
            updated = current_memory.model_copy(deep=True)
            updated.chapters.append(entry)
            updated.total_word_count += draft.word_count

            # Add repetition warnings from key claims
            for claim in entry.key_claims:
                warning = f"Chapter {draft.chapter_number} already covered: {claim}"
                if warning not in updated.repetition_warnings:
                    updated.repetition_warnings.append(warning)

            # Update token count estimate
            updated.token_count_estimate = count_tokens(
                updated.model_dump_json()
            )

            # Memory compression: if memory exceeds budget, compress older entries
            budget = settings.context_budget.dynamic_memory
            if updated.token_count_estimate > budget * 2:
                updated = _compress_memory(updated)

            logger.info(
                f"Memory updated: {len(updated.chapters)} chapters, "
                f"~{updated.token_count_estimate} tokens"
            )
            return updated

        except Exception as e:
            logger.error(f"Memory update attempt {attempt} failed: {e}")
            if attempt < settings.retry.max_validation_retries:
                time.sleep(settings.retry.retry_delay_seconds)
            else:
                # Fallback: create a minimal entry without LLM
                logger.warning("Memory update failed — using fallback minimal entry")
                fallback_entry = ChapterMemoryEntry(
                    chapter_number=draft.chapter_number,
                    title=draft.title,
                    summary=f"Chapter {draft.chapter_number}: {draft.title}. "
                            f"(Auto-summary unavailable — {draft.word_count} words)",
                )
                updated = current_memory.model_copy(deep=True)
                updated.chapters.append(fallback_entry)
                updated.total_word_count += draft.word_count
                return updated

    return current_memory  # Should not reach here


def _compress_memory(memory: DynamicMemory) -> DynamicMemory:
    """
    Compress older memory entries to stay within token budget.
    Strategy: Keep the last 3 chapters detailed, compress earlier ones to 1-sentence summaries.
    """
    logger = get_logger()
    if len(memory.chapters) <= 3:
        return memory

    compressed = memory.model_copy(deep=True)
    for i in range(len(compressed.chapters) - 3):
        entry = compressed.chapters[i]
        # Keep only the summary, clear detailed fields
        entry.key_claims = entry.key_claims[:2]  # Keep top 2 claims
        entry.definitions_introduced = []
        entry.characters_or_concepts = entry.characters_or_concepts[:2]
        entry.open_loops = [l for l in entry.open_loops if "unresolved" in l.lower()]
        entry.research_facts_used = []

    compressed.token_count_estimate = count_tokens(compressed.model_dump_json())
    logger.info(f"Memory compressed: ~{compressed.token_count_estimate} tokens")
    return compressed
```

---

## 2.6 — Assembler (Markdown Only)

### File: `auto_book/agents/assembler.py`

The Assembler merges all accepted chapters into a single Markdown file. This is a deterministic component — no LLM calls.

```python
"""
Assembler — merges accepted chapters into a final Markdown book.

No LLM calls. Pure deterministic text assembly.
"""

from pathlib import Path
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.utils.logger import get_logger


def assemble_markdown(
    book_bible: BookBible,
    chapters: list[ChapterDraft],
    output_dir: str,
) -> str:
    """
    Assemble all chapters into a single Markdown file.

    Args:
        book_bible: The Book Bible for title/metadata.
        chapters: List of accepted chapter drafts in order.
        output_dir: Directory to write the output file.

    Returns:
        The file path of the generated Markdown file.
    """
    logger = get_logger()
    output_path = Path(output_dir) / "book.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []

    # Title page
    parts.append(f"# {book_bible.working_title}\n")
    if book_bible.subtitle:
        parts.append(f"### {book_bible.subtitle}\n")
    parts.append("---\n")

    # Table of contents
    parts.append("## Table of Contents\n")
    for ch in chapters:
        anchor = ch.title.lower().replace(" ", "-").replace("'", "").replace('"', '')
        parts.append(f"- [Chapter {ch.chapter_number}: {ch.title}](#{anchor})")
    parts.append("\n---\n")

    # Chapters
    for ch in sorted(chapters, key=lambda c: c.chapter_number):
        parts.append(f"\n## Chapter {ch.chapter_number}: {ch.title}\n")
        parts.append(ch.body)
        if ch.key_takeaways:
            parts.append("\n### Key Takeaways\n")
            for takeaway in ch.key_takeaways:
                parts.append(f"- {takeaway}")
        parts.append("\n---\n")

    content = "\n".join(parts)
    output_path.write_text(content, encoding="utf-8")

    word_count = len(content.split())
    logger.info(f"Markdown assembled: {output_path} ({word_count} words, {len(chapters)} chapters)")

    return str(output_path)
```

---

## 2.7 — Orchestrator Updates

### Updated `orchestrator/graph.py`

Update the remaining node stubs to call real agents. Key changes:

1. **`review_chapter`** → calls `run_reviewer()`
2. **`update_memory`** → calls `run_memory_update()`
3. **`check_next`** → checks `current_chapter` against total chapters
4. **`assemble_book`** → calls `assemble_markdown()`
5. **`export_book`** → returns the Markdown path (DOCX added in Phase 3)
6. **Add checkpointing** — after every major node, save `RunState` to disk

### Rate Limiting

Add a rate-limit-aware wrapper around LLM calls. Groq's free tier has strict TPM limits.

Create `auto_book/utils/rate_limiter.py`:

```python
"""Simple rate limiter for Groq API calls."""

import time
from auto_book.utils.logger import get_logger

# Track last call time to enforce minimum delay
_last_call_time: float = 0.0
_MIN_DELAY_SECONDS: float = 3.0  # Minimum 3s between calls on free tier


def wait_for_rate_limit():
    """
    Block until enough time has passed since the last LLM call.
    Prevents hitting Groq's TPM/RPM limits.
    """
    global _last_call_time
    logger = get_logger()
    
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_DELAY_SECONDS:
        wait = _MIN_DELAY_SECONDS - elapsed
        logger.debug(f"Rate limiter: waiting {wait:.1f}s")
        time.sleep(wait)
    
    _last_call_time = time.time()
```

Call `wait_for_rate_limit()` at the start of every agent's LLM invocation (in `run_planner`, `run_writer`, `run_reviewer`, `run_memory_update`).

### Intermediate Artifact Saving

After each accepted chapter, save the chapter draft and review to individual files:

```python
# In orchestrator, after update_memory:
# Save chapter draft
chapter_path = Path(output_dir) / "chapters" / f"chapter_{ch_num:02d}.md"
chapter_path.parent.mkdir(parents=True, exist_ok=True)
chapter_path.write_text(draft.body, encoding="utf-8")

# Save review
review_path = Path(output_dir) / "reviews" / f"chapter_{ch_num:02d}_review.json"
review_path.parent.mkdir(parents=True, exist_ok=True)
review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")

# Save memory snapshot
memory_path = Path(output_dir) / "dynamic_memory.json"
memory_path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")

# Save Book Bible
bible_path = Path(output_dir) / "book_bible.json"
bible_path.write_text(bible.model_dump_json(indent=2), encoding="utf-8")
```

---

## 2.8 — Token Usage Tracking

Update the `TokenUsage` tracking in the orchestrator. After every LLM call, read the response metadata to track actual token usage:

```python
# Example in any agent after an LLM call:
# The ChatGroq response object has usage_metadata
# Access via: response.usage_metadata["input_tokens"], response.usage_metadata["output_tokens"]

# Update in orchestrator state:
token_usage = state.get("token_usage", TokenUsage())
token_usage.total_input_tokens += input_tokens
token_usage.total_output_tokens += output_tokens
token_usage.calls_made += 1
```

---

## 2.9 — Verification Checklist

After completing Phase 2, verify:

- [ ] `python -m auto_book "A beginner's guide to personal finance"` produces a complete Markdown book.
- [ ] The Book Bible is saved as `output/book_bible.json` and contains all required fields.
- [ ] Individual chapter files are saved under `output/chapters/`.
- [ ] Review decisions are saved under `output/reviews/`.
- [ ] `output/dynamic_memory.json` is updated after each chapter.
- [ ] `output/run_state.json` is saved after each step.
- [ ] The review loop correctly retries on `revise` (up to 2 times) and `fail` (up to 1 time).
- [ ] The final `output/book.md` contains all chapters in order with a table of contents.
- [ ] Run logs in `output/run.log` show every agent call with token counts.
- [ ] Rate limiting prevents hitting Groq's TPM limits (no 429 errors).
- [ ] Memory compression kicks in if memory grows beyond 2x the budget.
- [ ] The system handles Groq API errors gracefully (retries, then fails with clear message).
- [ ] Running `--resume` after a crash resumes from the last completed chapter.

### Quality Check

Read the generated book and verify:
- Chapters follow a logical progression.
- No major repetition between chapters.
- Tone is consistent throughout.
- The book delivers on the promise from the Book Bible.
- Word count per chapter is reasonable (800-2500 words).
