"""Recall tool — look up a problem, experience, or card by ID."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.storage.fs_backend import FSBackend
from agenix.tools.base import error_result, text_result

_VALID_TYPES = {"problem", "experience", "card"}


def create_tool(*, fs_backend: FSBackend) -> SdkMcpTool[Any]:
    """Create a recall MCP tool backed by the given FSBackend."""

    @tool(
        "recall",
        "Look up the raw content of a problem, experience, or card by its ID",
        {
            "entity_type": str,
            "entity_id": str,
        },
    )
    async def recall(args: dict) -> dict:
        entity_type = args.get("entity_type", "")
        entity_id = args.get("entity_id", "")

        if not entity_type:
            return error_result("entity_type is required")
        if not entity_id:
            return error_result("entity_id is required")
        if entity_type not in _VALID_TYPES:
            return error_result(
                f"entity_type must be one of: {', '.join(sorted(_VALID_TYPES))}"
            )

        if entity_type == "problem":
            entity = fs_backend.get_problem(entity_id)
        elif entity_type == "experience":
            entity = fs_backend.get_experience(entity_id)
        else:
            entity = fs_backend.get_card(entity_id)

        if entity is None:
            return error_result(f"{entity_type} '{entity_id}' not found")

        return text_result(json.dumps({
            "entity_type": entity_type,
            "entity_id": entity_id,
            "data": json.loads(entity.model_dump_json()),
        }, indent=2))

    return recall
