"""Pydantic data models for the reflection system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from ulid import ULID


def _ulid() -> str:
    return str(ULID())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- Problem ---


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ProblemStatus(str, Enum):
    PROPOSED = "proposed"
    SOLVING = "solving"
    SOLVED = "solved"
    FAILED = "failed"


class TestCase(BaseModel):
    input: str
    expected_output: str
    description: str = ""


class Problem(BaseModel):
    problem_id: str = Field(default_factory=_ulid)
    title: str
    description: str
    test_cases: list[TestCase] = Field(default_factory=list)
    domain: str = "general"
    difficulty: Difficulty = Difficulty.MEDIUM
    status: ProblemStatus = ProblemStatus.PROPOSED
    created_at: datetime = Field(default_factory=_now)


# --- Trajectory ---


class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FEEDBACK = "feedback"


class TrajectoryStep(BaseModel):
    step_index: int
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    timestamp: datetime = Field(default_factory=_now)


class TestResult(BaseModel):
    test_case: TestCase
    passed: bool
    actual_output: str = ""
    error: str = ""


class Trajectory(BaseModel):
    trajectory_id: str = Field(default_factory=_ulid)
    problem_id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: str = ""
    code_solution: str = ""
    is_correct: bool = False
    test_results: list[TestResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = None


# --- Cards ---


class CardType(str, Enum):
    KNOWLEDGE = "knowledge"
    INSIGHT = "insight"
    REFLECTION = "reflection"


class ReflectionCategory(str, Enum):
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    PATTERN = "pattern"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    GENERAL = "general"


class CardStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class LineageOperation(str, Enum):
    CREATE = "create"
    REVISE = "revise"
    MERGE = "merge"
    SPLIT = "split"
    SUPERSEDE = "supersede"
    ARCHIVE = "archive"


class SourceReference(BaseModel):
    """Typed reference to a source entity."""

    id: str
    type: str  # "trajectory" | "reflection" | "card"


class LineageEvent(BaseModel):
    """Immutable record of a lineage operation."""

    operation: LineageOperation
    timestamp: datetime = Field(default_factory=_now)
    agent: str = ""
    run_tag: str = ""
    source_refs: list[SourceReference] = Field(default_factory=list)
    from_version: Optional[int] = None
    merged_card_ids: list[str] = Field(default_factory=list)
    split_from_card_id: Optional[str] = None
    superseded_by: Optional[str] = None
    description: str = ""


class Card(BaseModel):
    card_id: str = Field(default_factory=_ulid)
    card_type: CardType
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    version: int = 1
    status: CardStatus = CardStatus.ACTIVE
    lineage: list[LineageEvent] = Field(default_factory=list)
    source_refs: list[SourceReference] = Field(default_factory=list)
    superseded_by: Optional[str] = None
    predecessor_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class KnowledgeCard(Card):
    card_type: CardType = CardType.KNOWLEDGE
    domain: str = "general"
    applicability: str = ""
    limitations: str = ""
    related_card_ids: list[str] = Field(default_factory=list)


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class InsightCard(Card):
    card_type: CardType = CardType.INSIGHT
    hypothesis: str = ""
    hypothesis_status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    experiments_run: int = 0


class ReflectionCard(Card):
    card_type: CardType = CardType.REFLECTION
    trajectory_id: str
    category: ReflectionCategory = ReflectionCategory.GENERAL
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_steps: list[int] = Field(default_factory=list)


# --- Agent Definition (loaded from files) ---


class AgentConfig(BaseModel):
    """Configuration loaded from an agent's config.toml."""

    model: str = "sonnet"
    temperature: float = 0.7
    max_turns: int = 10
    tools: list[str] = Field(default_factory=list)
    custom_tools: list[str] = Field(default_factory=list)


class LoadedAgent(BaseModel):
    """An agent definition loaded from the agents/<name>/<variant>/ directory."""

    name: str
    variant: str = "base"
    description: str = ""
    system_prompt: str = ""
    input_format: str = ""
    output_format: str = ""
    examples: str = ""
    config: AgentConfig = Field(default_factory=AgentConfig)
    logic_module_path: Optional[str] = None
