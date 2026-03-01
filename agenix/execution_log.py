"""Persistent structured execution logging for the agenix engine.

Captures orchestration steps (message lifecycle, agent invocations,
knowledge retrieval, output parsing, etc.) as append-only JSONL.
Separate from trajectories which record agent-LLM conversation turns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from ulid import ULID

logger = logging.getLogger(__name__)


def _ulid() -> str:
    return str(ULID())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventType(str, Enum):
    """Types of execution events."""

    # Loop lifecycle
    LOOP_STARTED = "loop_started"
    LOOP_STOPPED = "loop_stopped"

    # Message lifecycle
    MESSAGE_DEQUEUED = "message_dequeued"
    MESSAGE_COMPLETED = "message_completed"
    MESSAGE_FAILED = "message_failed"

    # Agent invocation
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"

    # Handler steps
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    OUTPUT_PARSED = "output_parsed"
    DATA_SAVED = "data_saved"
    MESSAGE_ENQUEUED = "message_enqueued"

    # Scheduled
    SCHEDULED_TRIGGER = "scheduled_trigger"

    # Errors
    HANDLER_ERROR = "handler_error"


class ExecutionEvent(BaseModel):
    """A single structured execution event."""

    event_id: str = Field(default_factory=_ulid)
    timestamp: datetime = Field(default_factory=_now)
    event_type: EventType
    agent: str = ""
    run_tag: str = ""
    message_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class ExecutionLogger:
    """Append-only structured execution logger.

    Writes ExecutionEvent records as JSONL to a run-scoped file.
    """

    def __init__(
        self,
        log_path: Path,
        run_tag: str = "",
        agent: str = "",
    ) -> None:
        self._path = log_path
        self._run_tag = run_tag
        self._agent = agent
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: ExecutionEvent) -> None:
        """Append a single event to the JSONL file."""
        if not event.run_tag:
            event.run_tag = self._run_tag
        if not event.agent:
            event.agent = self._agent
        line = event.model_dump_json() + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    # --- Convenience methods ---

    def loop_started(self, loop_type: str, **kwargs: Any) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.LOOP_STARTED,
            data={"loop_type": loop_type, **kwargs},
        ))

    def loop_stopped(self) -> None:
        self.emit(ExecutionEvent(event_type=EventType.LOOP_STOPPED))

    def message_dequeued(
        self, message_id: str, queue_name: str, payload: dict,
    ) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.MESSAGE_DEQUEUED,
            message_id=message_id,
            data={"queue_name": queue_name, "payload_keys": list(payload.keys())},
        ))

    def message_completed(self, message_id: str) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.MESSAGE_COMPLETED,
            message_id=message_id,
        ))

    def message_failed(
        self, message_id: str, error: str, error_type: str = "",
    ) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.MESSAGE_FAILED,
            message_id=message_id,
            error=error,
            error_type=error_type,
        ))

    def knowledge_retrieval(
        self, query: str, num_hits: int, limit: int,
    ) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.KNOWLEDGE_RETRIEVAL,
            data={"query": query[:200], "num_hits": num_hits, "limit": limit},
        ))

    def agent_started(
        self,
        agent_name: str,
        model: str,
        max_turns: int,
        input_size_chars: int,
    ) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.AGENT_STARTED,
            data={
                "agent_name": agent_name,
                "model": model,
                "max_turns": max_turns,
                "input_size_chars": input_size_chars,
            },
        ))

    def agent_completed(
        self,
        agent_name: str,
        duration_ms: int,
        num_turns: int,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.AGENT_COMPLETED,
            duration_ms=duration_ms,
            data={
                "agent_name": agent_name,
                "num_turns": num_turns,
                "cost_usd": cost_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        ))

    def output_parsed(
        self,
        parser: str,
        success: bool,
        entities: list[str] | None = None,
        error: str = "",
    ) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.OUTPUT_PARSED,
            data={
                "parser": parser,
                "success": success,
                "entities": entities or [],
            },
            error=error if not success else None,
        ))

    def data_saved(self, entity_type: str, entity_id: str) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.DATA_SAVED,
            data={"entity_type": entity_type, "entity_id": entity_id},
        ))

    def message_enqueued(self, queue_name: str, message_id: str) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.MESSAGE_ENQUEUED,
            data={"queue_name": queue_name, "message_id": message_id},
        ))

    def scheduled_trigger(self, handler_name: str) -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.SCHEDULED_TRIGGER,
            data={"handler_name": handler_name},
        ))

    def handler_error(self, error: str, error_type: str = "") -> None:
        self.emit(ExecutionEvent(
            event_type=EventType.HANDLER_ERROR,
            error=error,
            error_type=error_type,
        ))


class NullExecutionLogger(ExecutionLogger):
    """No-op logger for when execution logging is not needed."""

    def __init__(self) -> None:
        # Don't call super().__init__ — no path, no dirs
        self._path = Path("/dev/null")
        self._run_tag = ""
        self._agent = ""

    def emit(self, event: ExecutionEvent) -> None:
        pass


def create_execution_logger(
    storage_config: Any,
    run_tag: str,
    agent: str = "",
) -> ExecutionLogger:
    """Create an ExecutionLogger for a given run.

    Args:
        storage_config: StorageConfig with execution_log_path method.
        run_tag: The run tag for this execution.
        agent: Default agent name for events.
    """
    log_path = storage_config.execution_log_path(run_tag)
    return ExecutionLogger(log_path=log_path, run_tag=run_tag, agent=agent)
