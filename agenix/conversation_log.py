"""Full LLM conversation logging as append-only JSONL.

Captures every message in the agent-LLM conversation (system, user/tool
results, assistant text/thinking/tool_use, final result metadata) so the
full interaction can be replayed and analysed later.

File convention:
    <reflection_root>/<env>/experiences/<agent_name>/<experience_id>.jsonl
Each line is one JSON object following standard LLM chat format.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_block(block: Any) -> dict:
    """Convert an SDK content block to a plain dict."""
    from claude_agent_sdk.types import (
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        d: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
        }
        if block.content is not None:
            d["content"] = block.content
        if block.is_error:
            d["is_error"] = True
        return d
    # Fallback: try dataclass fields
    if hasattr(block, "__dataclass_fields__"):
        from dataclasses import asdict

        return {"type": type(block).__name__, **asdict(block)}
    return {"type": type(block).__name__, "repr": repr(block)}


class ConversationLogger:
    """Append-only JSONL writer for a single agent conversation."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._turn = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def turn(self) -> int:
        return self._turn

    def _append(self, record: dict) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def log_system(self, msg: Any) -> None:
        """Log a SystemMessage."""
        self._append({
            "role": "system",
            "subtype": msg.subtype,
            "content": msg.data,
            "timestamp": _now_iso(),
            "turn": self._turn,
        })

    def log_user(self, msg: Any) -> None:
        """Log a UserMessage.

        Plain-text content → single 'user' line.
        List content (tool results) → one 'tool' line per ToolResultBlock.
        """
        from claude_agent_sdk.types import ToolResultBlock

        if isinstance(msg.content, str):
            self._append({
                "role": "user",
                "content": msg.content,
                "timestamp": _now_iso(),
                "turn": self._turn,
            })
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    self._append({
                        "role": "tool",
                        "content": block.content,
                        "tool_call_id": block.tool_use_id,
                        "is_error": bool(block.is_error),
                        "timestamp": _now_iso(),
                        "turn": self._turn,
                    })
                else:
                    self._append({
                        "role": "user",
                        "content": _serialize_block(block),
                        "timestamp": _now_iso(),
                        "turn": self._turn,
                    })

    def log_assistant(self, msg: Any) -> None:
        """Log an AssistantMessage.

        Increments the turn counter. Emits full content array and an
        OpenAI-compatible tool_calls array for tool_use blocks.
        """
        self._turn += 1
        content = [_serialize_block(b) for b in msg.content]

        # Build OpenAI-style tool_calls array
        from claude_agent_sdk.types import ToolUseBlock

        tool_calls = []
        for b in msg.content:
            if isinstance(b, ToolUseBlock):
                tool_calls.append({
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.input, default=str),
                    },
                })

        record: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "model": msg.model,
            "timestamp": _now_iso(),
            "turn": self._turn,
        }
        if tool_calls:
            record["tool_calls"] = tool_calls
        if msg.error:
            record["error"] = str(msg.error)

        self._append(record)

    def log_result(self, msg: Any) -> None:
        """Log a ResultMessage — cost, tokens, duration metadata."""
        self._append({
            "role": "result",
            "duration_ms": msg.duration_ms,
            "num_turns": msg.num_turns,
            "total_cost_usd": msg.total_cost_usd,
            "usage": msg.usage,
            "is_error": msg.is_error,
            "result": msg.result,
            "session_id": msg.session_id,
            "timestamp": _now_iso(),
            "turn": self._turn,
        })


class NullConversationLogger(ConversationLogger):
    """No-op logger for when conversation logging is not needed."""

    def __init__(self) -> None:
        self._path = Path("/dev/null")
        self._turn = 0

    def _append(self, record: dict) -> None:
        pass
