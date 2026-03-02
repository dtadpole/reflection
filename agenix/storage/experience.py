"""Experience convenience helpers built on the filesystem backend."""

from __future__ import annotations

import json


def extract_problem_id(log_text: str) -> str | None:
    """Extract problem_id from the first user message in a .jsonl experience log."""
    for line in log_text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("role") == "user":
            content = entry.get("content", "")
            try:
                payload = json.loads(content)
                return payload.get("problem", {}).get("problem_id")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
    return None
