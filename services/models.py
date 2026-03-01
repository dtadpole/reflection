"""Data models for remote services."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class KernelExecResult(BaseModel):
    """Result of a kernel evaluation (compilation, correctness, timing)."""

    compiled: bool = False
    correctness: bool = False
    runtime: float = -1.0  # milliseconds
    metadata: dict = Field(default_factory=dict)
    runtime_stats: dict = Field(default_factory=dict)


class CorrectnessResult(BaseModel):
    """Detailed correctness verification result."""

    total_trials: int = 0
    passed_trials: int = 0
    max_diff: float = -1.0
    avg_diff: float = -1.0


class ServiceStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


class ServiceHealth(BaseModel):
    """Health status of a remote service endpoint."""

    name: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    endpoint: str = ""
    devices: list[str] = Field(default_factory=list)
    pending_requests: int = 0
    checked_at: datetime = Field(default_factory=_now)
