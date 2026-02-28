"""Filesystem-based message queue with atomic state transitions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agenix.config import StorageConfig
from agenix.queue.models import MessageState, QueueMessage

logger = logging.getLogger(__name__)


class FSQueue:
    """A filesystem-based FIFO message queue.

    Messages are stored as JSON files. State is determined by subdirectory
    (pending/, processing/, done/, failed/). State transitions use os.rename()
    for atomicity on POSIX filesystems.

    FIFO ordering relies on ULID message IDs (lexicographic = chronological).
    """

    STATES = (
        MessageState.PENDING,
        MessageState.PROCESSING,
        MessageState.DONE,
        MessageState.FAILED,
    )

    def __init__(self, queue_name: str, config: StorageConfig) -> None:
        self.queue_name = queue_name
        self._root = config.queues_path / queue_name

    def _state_dir(self, state: MessageState) -> Path:
        return self._root / state.value

    def _message_path(self, state: MessageState, message_id: str) -> Path:
        return self._state_dir(state) / f"{message_id}.json"

    def initialize(self) -> None:
        """Create queue directories if they don't exist."""
        for state in self.STATES:
            self._state_dir(state).mkdir(parents=True, exist_ok=True)

    # --- Core operations ---

    def enqueue(self, sender: str, payload: dict) -> QueueMessage:
        """Create a new message in the pending directory."""
        msg = QueueMessage(
            queue_name=self.queue_name,
            sender=sender,
            payload=payload,
        )
        path = self._message_path(MessageState.PENDING, msg.message_id)
        path.write_text(json.dumps(msg.model_dump(mode="json"), indent=2))
        logger.debug("Enqueued message %s to %s", msg.message_id, self.queue_name)
        return msg

    def dequeue(self) -> Optional[QueueMessage]:
        """Atomically claim the oldest pending message.

        Moves the file from pending/ to processing/ via os.rename(). If the
        rename fails (another process claimed it), tries the next message.
        Returns None if no pending messages are available.
        """
        pending_dir = self._state_dir(MessageState.PENDING)
        if not pending_dir.exists():
            return None

        for path in sorted(pending_dir.iterdir()):
            if not path.name.endswith(".json"):
                continue
            message_id = path.stem
            dest = self._message_path(MessageState.PROCESSING, message_id)
            try:
                path.rename(dest)
            except OSError:
                # Another process claimed this message
                continue

            # Rename succeeded — we own this message. Update claimed_at.
            msg = QueueMessage.model_validate_json(dest.read_text())
            msg.claimed_at = datetime.now(timezone.utc)
            dest.write_text(json.dumps(msg.model_dump(mode="json"), indent=2))
            logger.debug("Dequeued message %s from %s", message_id, self.queue_name)
            return msg

        return None

    def complete(self, message_id: str) -> QueueMessage:
        """Move a processing message to done."""
        src = self._message_path(MessageState.PROCESSING, message_id)
        dest = self._message_path(MessageState.DONE, message_id)
        src.rename(dest)

        msg = QueueMessage.model_validate_json(dest.read_text())
        msg.completed_at = datetime.now(timezone.utc)
        dest.write_text(json.dumps(msg.model_dump(mode="json"), indent=2))
        logger.debug("Completed message %s in %s", message_id, self.queue_name)
        return msg

    def fail(self, message_id: str, error: str) -> QueueMessage:
        """Move a processing message to failed with an error reason."""
        src = self._message_path(MessageState.PROCESSING, message_id)
        dest = self._message_path(MessageState.FAILED, message_id)
        src.rename(dest)

        msg = QueueMessage.model_validate_json(dest.read_text())
        msg.completed_at = datetime.now(timezone.utc)
        msg.error = error
        dest.write_text(json.dumps(msg.model_dump(mode="json"), indent=2))
        logger.debug("Failed message %s in %s: %s", message_id, self.queue_name, error)
        return msg

    # --- Query operations ---

    def peek(self) -> Optional[QueueMessage]:
        """Read the oldest pending message without claiming it."""
        pending_dir = self._state_dir(MessageState.PENDING)
        if not pending_dir.exists():
            return None

        for path in sorted(pending_dir.iterdir()):
            if not path.name.endswith(".json"):
                continue
            return QueueMessage.model_validate_json(path.read_text())

        return None

    def _list_state(self, state: MessageState, limit: int = 100) -> list[QueueMessage]:
        state_dir = self._state_dir(state)
        if not state_dir.exists():
            return []
        messages = []
        for path in sorted(state_dir.iterdir()):
            if not path.name.endswith(".json"):
                continue
            messages.append(QueueMessage.model_validate_json(path.read_text()))
            if len(messages) >= limit:
                break
        return messages

    def list_pending(self, limit: int = 100) -> list[QueueMessage]:
        return self._list_state(MessageState.PENDING, limit)

    def list_processing(self, limit: int = 100) -> list[QueueMessage]:
        return self._list_state(MessageState.PROCESSING, limit)

    def list_failed(self, limit: int = 100) -> list[QueueMessage]:
        return self._list_state(MessageState.FAILED, limit)

    def count(self, state: MessageState) -> int:
        state_dir = self._state_dir(state)
        if not state_dir.exists():
            return 0
        return sum(1 for p in state_dir.iterdir() if p.name.endswith(".json"))
