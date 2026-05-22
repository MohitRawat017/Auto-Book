"""LangGraph state machine for the Phase 1 skeleton."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from auto_book.agents.assembler import assemble_stub
from auto_book.agents.memory_updater import run_memory_update
from auto_book.agents.planner import run_planner
from auto_book.agents.reviewer import run_reviewer
from auto_book.agents.writer import run_writer
from auto_book.config import settings
from auto_book.models.chapter import ChapterPlan
from auto_book.models.memory import DynamicMemory
from auto_book.models.review import ReviewDecisionEnum
from auto_book.models.run_state import ChapterStatus, RunPhase
from auto_book.orchestrator.checkpointer import save_graph_checkpoint
from auto_book.orchestrator.state import GraphState
from auto_book.utils.logger import get_logger


def _checkpointing_node(node: Callable[[GraphState], dict[str, Any]]):
    """Wrap a node so each completed step is checkpointed."""

    def wrapped(state: GraphState) -> dict[str, Any]:
        update = node(state)
        merged = dict(state)
        merged.update(update)
        save_graph_checkpoint(merged)
        return update

    wrapped.__name__ = node.__name__
    return wrapped


def _update_chapter_status(
    statuses: list[ChapterStatus],
    chapter_number: int,
    **updates: Any,
) -> list[ChapterStatus]:
    """Return chapter statuses with one chapter updated or appended."""

    updated_statuses: list[ChapterStatus] = []
    found = False
    for status in statuses:
        if status.chapter_number == chapter_number:
            updated_statuses.append(status.model_copy(update=updates))
            found = True
        else:
            updated_statuses.append(status)

    if not found:
        updated_statuses.append(
            ChapterStatus(chapter_number=chapter_number).model_copy(update=updates)
        )

    return updated_statuses


def collect_input(state: GraphState) -> dict[str, Any]:
    """Validate and prepare user input."""

    logger = get_logger()
    brief = (state.get("user_brief") or "").strip()
    if not brief:
        return {"phase": RunPhase.FAILED, "error": "User brief is required."}

    logger.info("Collecting input for brief: %s", brief[:100])
    return {
        "user_brief": brief,
        "genre": state.get("genre") or "non-fiction how-to",
        "phase": RunPhase.INITIALIZED,
    }


def plan_book(state: GraphState) -> dict[str, Any]:
    """Create a stub Book Bible."""

    logger = get_logger()
    if state.get("error"):
        return {"phase": RunPhase.FAILED}

    bible = run_planner(state["user_brief"], state.get("genre", "non-fiction how-to"))
    chapter_statuses = [
        ChapterStatus(chapter_number=chapter.chapter_number)
        for chapter in bible.chapter_outline
    ]
    logger.info("Planning complete: %s", bible.working_title)
    return {
        "book_bible": bible,
        "chapter_statuses": chapter_statuses,
        "current_chapter": 0,
        "phase": RunPhase.PLANNING,
    }


def prepare_chapter(state: GraphState) -> dict[str, Any]:
    """Extract the next chapter plan from the Book Bible."""

    logger = get_logger()
    bible = state.get("book_bible")
    if bible is None:
        return {"phase": RunPhase.FAILED, "error": "Book Bible is missing."}

    current = int(state.get("current_chapter") or 0) + 1
    if current > len(bible.chapter_outline):
        return {"phase": RunPhase.ASSEMBLING}

    outline = bible.chapter_outline[current - 1]
    chapter_plan = ChapterPlan(
        chapter_number=outline.chapter_number,
        title=outline.title,
        summary=outline.summary,
        key_topics=outline.key_topics,
        word_count_target=outline.word_count_target,
        continuity_notes=(
            f"{len(state.get('dynamic_memory', DynamicMemory()).chapters)} "
            "chapter(s) accepted so far."
        ),
    )
    statuses = _update_chapter_status(
        state.get("chapter_statuses", []),
        current,
        status="drafting",
        revision_count=0,
        regeneration_count=0,
        current_draft=None,
        last_review=None,
    )
    logger.info("Preparing chapter %s: %s", current, chapter_plan.title)
    return {
        "current_chapter": current,
        "chapter_plan": chapter_plan,
        "chapter_draft": None,
        "review_decision": None,
        "revision_count": 0,
        "regeneration_count": 0,
        "chapter_statuses": statuses,
        "phase": RunPhase.WRITING,
    }


def write_chapter(state: GraphState) -> dict[str, Any]:
    """Create a stub chapter draft."""

    logger = get_logger()
    bible = state.get("book_bible")
    chapter_plan = state.get("chapter_plan")
    if bible is None or chapter_plan is None:
        return {
            "phase": RunPhase.FAILED,
            "error": "Cannot write chapter without Book Bible and ChapterPlan.",
        }

    review = state.get("review_decision")
    revision_count = int(state.get("revision_count") or 0)
    regeneration_count = int(state.get("regeneration_count") or 0)
    revision_feedback = None
    if review and review.decision == ReviewDecisionEnum.REVISE:
        revision_count += 1
        revision_feedback = review.required_fixes
    elif review and review.decision == ReviewDecisionEnum.FAIL:
        regeneration_count += 1

    draft = run_writer(
        chapter_plan=chapter_plan,
        book_bible=bible,
        dynamic_memory=state.get("dynamic_memory", DynamicMemory()),
        revision_feedback=revision_feedback,
    )
    statuses = _update_chapter_status(
        state.get("chapter_statuses", []),
        draft.chapter_number,
        status="drafted",
        revision_count=revision_count,
        regeneration_count=regeneration_count,
        current_draft=draft,
    )
    logger.info("Writing complete for chapter %s", draft.chapter_number)
    return {
        "chapter_draft": draft,
        "revision_count": revision_count,
        "regeneration_count": regeneration_count,
        "chapter_statuses": statuses,
        "phase": RunPhase.WRITING,
    }


def review_chapter(state: GraphState) -> dict[str, Any]:
    """Review the current stub draft."""

    logger = get_logger()
    bible = state.get("book_bible")
    draft = state.get("chapter_draft")
    if bible is None or draft is None:
        return {
            "phase": RunPhase.FAILED,
            "error": "Cannot review chapter without Book Bible and ChapterDraft.",
        }

    review = run_reviewer(draft, bible, state.get("dynamic_memory", DynamicMemory()))
    statuses = _update_chapter_status(
        state.get("chapter_statuses", []),
        draft.chapter_number,
        status="reviewed",
        last_review=review,
    )
    logger.info(
        "Review complete for chapter %s: %s",
        draft.chapter_number,
        review.decision.value,
    )
    return {
        "review_decision": review,
        "chapter_statuses": statuses,
        "phase": RunPhase.REVIEWING,
    }


def update_memory(state: GraphState) -> dict[str, Any]:
    """Update memory and accept a passing chapter."""

    logger = get_logger()
    draft = state.get("chapter_draft")
    if draft is None:
        return {"phase": RunPhase.FAILED, "error": "No draft to accept."}

    memory = run_memory_update(draft, state.get("dynamic_memory", DynamicMemory()))
    statuses = _update_chapter_status(
        state.get("chapter_statuses", []),
        draft.chapter_number,
        status="accepted",
        accepted_draft=draft,
        current_draft=draft,
    )
    logger.info("Chapter %s accepted and memory updated", draft.chapter_number)
    return {
        "dynamic_memory": memory,
        "chapter_statuses": statuses,
        "phase": RunPhase.MEMORY_UPDATE,
    }


def check_next(state: GraphState) -> dict[str, Any]:
    """Record progress before routing to the next chapter or assembly."""

    bible = state.get("book_bible")
    total = len(bible.chapter_outline) if bible else settings.book.default_chapter_count
    current = int(state.get("current_chapter") or 0)
    get_logger().info("Progress check: chapter %s/%s", current, total)
    return {}


def assemble_book(state: GraphState) -> dict[str, Any]:
    """Run the Phase 1 stub assembler."""

    logger = get_logger()
    bible = state.get("book_bible")
    if bible is None:
        return {"phase": RunPhase.FAILED, "error": "No Book Bible to assemble."}

    accepted = [
        status.accepted_draft
        for status in state.get("chapter_statuses", [])
        if status.accepted_draft is not None
    ]
    export_result = assemble_stub(bible, accepted)
    logger.info("Stub assembly complete")
    return {
        "export_result": export_result,
        "phase": RunPhase.ASSEMBLING,
    }


def export_book(state: GraphState) -> dict[str, Any]:
    """Complete the Phase 1 placeholder export step."""

    get_logger().info("Phase 1 export step complete (no files written).")
    return {"phase": RunPhase.COMPLETED}


def handle_failure(state: GraphState) -> dict[str, Any]:
    """Handle unrecoverable failures."""

    error = state.get("error") or "Unknown error"
    get_logger().error("Run failed: %s", error)
    return {"phase": RunPhase.FAILED, "error": error}


def route_after_review(state: GraphState) -> str:
    """Route after reviewer output."""

    if state.get("phase") == RunPhase.FAILED:
        return "handle_failure"

    review = state.get("review_decision")
    if review is None or review.decision == ReviewDecisionEnum.PASS:
        return "update_memory"

    revision_count = int(state.get("revision_count") or 0)
    regeneration_count = int(state.get("regeneration_count") or 0)

    if review.decision == ReviewDecisionEnum.REVISE:
        if revision_count < settings.retry.max_revisions:
            return "write_chapter"
        return "handle_failure"

    if review.decision == ReviewDecisionEnum.FAIL:
        if regeneration_count < settings.retry.max_regenerations:
            return "write_chapter"
        return "handle_failure"

    return "handle_failure"


def route_after_check(state: GraphState) -> str:
    """Route to the next chapter or assembly."""

    bible = state.get("book_bible")
    total = len(bible.chapter_outline) if bible else settings.book.default_chapter_count
    current = int(state.get("current_chapter") or 0)
    return "prepare_chapter" if current < total else "assemble_book"


def build_graph():
    """Construct and compile the Phase 1 graph."""

    graph = StateGraph(GraphState)

    graph.add_node("collect_input", _checkpointing_node(collect_input))
    graph.add_node("plan_book", _checkpointing_node(plan_book))
    graph.add_node("prepare_chapter", _checkpointing_node(prepare_chapter))
    graph.add_node("write_chapter", _checkpointing_node(write_chapter))
    graph.add_node("review_chapter", _checkpointing_node(review_chapter))
    graph.add_node("update_memory", _checkpointing_node(update_memory))
    graph.add_node("check_next", _checkpointing_node(check_next))
    graph.add_node("assemble_book", _checkpointing_node(assemble_book))
    graph.add_node("export_book", _checkpointing_node(export_book))
    graph.add_node("handle_failure", _checkpointing_node(handle_failure))

    graph.set_entry_point("collect_input")
    graph.add_edge("collect_input", "plan_book")
    graph.add_edge("plan_book", "prepare_chapter")
    graph.add_edge("prepare_chapter", "write_chapter")
    graph.add_edge("write_chapter", "review_chapter")
    graph.add_edge("update_memory", "check_next")
    graph.add_edge("assemble_book", "export_book")
    graph.add_edge("export_book", END)
    graph.add_edge("handle_failure", END)

    graph.add_conditional_edges(
        "review_chapter",
        route_after_review,
        {
            "update_memory": "update_memory",
            "write_chapter": "write_chapter",
            "handle_failure": "handle_failure",
        },
    )
    graph.add_conditional_edges(
        "check_next",
        route_after_check,
        {
            "prepare_chapter": "prepare_chapter",
            "assemble_book": "assemble_book",
        },
    )

    return graph.compile()
