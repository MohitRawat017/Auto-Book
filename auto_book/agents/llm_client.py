"""Centralized LLM client factory for live Phase 2 agents."""

from langchain_groq import ChatGroq

from auto_book.config import require_groq_api_key, settings


def get_llm(role: str) -> ChatGroq:
    """Return a configured ChatGroq client for an agent role."""

    model_map = {
        "writer": settings.llm.writer_model,
        "planner": settings.llm.planner_model,
        "reviewer": settings.llm.reviewer_model,
        "memory": settings.llm.memory_model,
    }
    temperature_map = {
        "writer": settings.llm.temperature_creative,
        "planner": settings.llm.temperature_creative,
        "reviewer": settings.llm.temperature_analytical,
        "memory": settings.llm.temperature_analytical,
    }

    return ChatGroq(
        api_key=require_groq_api_key(),
        model=model_map.get(role, settings.llm.writer_model),
        temperature=temperature_map.get(role, 0.5),
    )
