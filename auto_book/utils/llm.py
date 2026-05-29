"""Shared LLM response parsing utilities."""

import json
from json import JSONDecodeError
from typing import Any


def extract_message_text(response: object) -> str:
    """Extract text content from a LangChain chat response."""
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


def strip_code_fence(text: str) -> str:
    """Remove wrapping ```...``` code fences from LLM output."""
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    return body


def load_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM text, tolerating surrounding prose."""
    try:
        data = json.loads(text)
    except JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object, got: " + type(data).__name__)
    return data
