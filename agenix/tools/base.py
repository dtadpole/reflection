"""Base tool helpers for agenix custom tools."""

from __future__ import annotations

from typing import Any


def text_result(text: str) -> dict[str, Any]:
    """Create a standard text result for an MCP tool response."""
    return {"content": [{"type": "text", "text": text}]}


def error_result(text: str) -> dict[str, Any]:
    """Create an error result for an MCP tool response."""
    return {"content": [{"type": "text", "text": text}], "is_error": True}
