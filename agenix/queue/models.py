"""Queue message models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from ulid import ULID


def _ulid() -> str:
    return str(ULID())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MessageState(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class QueueMessage(BaseModel):
    message_id: str = Field(default_factory=_ulid)
    queue_name: str
    sender: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=_now)
    claimed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
