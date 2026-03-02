"""Recall MCP tools — recall (entity lookup) and excerpt (JSONL row-range reader)."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.storage.fs_backend import FSBackend
from agenix.tools.base import error_result, text_result

_VALID_TYPES = {"problem", "experience", "card"}


def create_tool(*, fs_backend: FSBackend) -> list[SdkMcpTool[Any]]:
    """Create recall and excerpt MCP tools backed by the given FSBackend."""

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
            if entity is None:
                return error_result(f"problem '{entity_id}' not found")
            return text_result(json.dumps({
                "entity_type": "problem",
                "entity_id": entity_id,
                "data": json.loads(entity.model_dump_json()),
            }, indent=2))

        if entity_type == "experience":
            log_text = fs_backend.get_experience_log(entity_id)
            if log_text is None:
                return error_result(f"experience '{entity_id}' not found")
            return text_result(json.dumps({
                "entity_type": "experience",
                "entity_id": entity_id,
                "format": "jsonl",
                "data": log_text,
            }, indent=2))

        # card
        entity = fs_backend.get_card(entity_id)
        if entity is None:
            return error_result(f"card '{entity_id}' not found")
        return text_result(json.dumps({
            "entity_type": "card",
            "entity_id": entity_id,
            "data": json.loads(entity.model_dump_json()),
        }, indent=2))

    @tool(
        "excerpt",
        "Read specific rows from a JSONL experience file by row range",
        {
            "experience_id": str,
            "start_row": int,
            "end_row": int,
        },
    )
    async def excerpt(args: dict) -> dict:
        experience_id = args.get("experience_id", "")
        if not experience_id:
            return error_result("experience_id is required")

        log_text = fs_backend.get_experience_log(experience_id)
        if log_text is None:
            return error_result(f"experience '{experience_id}' not found")

        all_lines = [ln for ln in log_text.splitlines() if ln.strip()]
        total_rows = len(all_lines)

        start_row = max(1, args.get("start_row") or 1)
        end_row = min(total_rows, args.get("end_row") or total_rows)

        if start_row > total_rows:
            return error_result(
                f"start_row {start_row} exceeds total rows ({total_rows})"
            )

        selected = all_lines[start_row - 1:end_row]

        rows = []
        for line in selected:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_raw": line})

        return text_result(json.dumps({
            "experience_id": experience_id,
            "start_row": start_row,
            "end_row": min(end_row, total_rows),
            "total_rows": total_rows,
            "rows": rows,
        }, indent=2))

    return [recall, excerpt]
