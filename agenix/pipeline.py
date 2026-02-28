"""Pipeline orchestrator for the reflection loop.

Runs agents in sequence: Curator -> Solver -> Critic -> Organizer,
with periodic Insight Finder runs. Each step builds input from
stored data, invokes the agent, and persists results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from agenix.config import ReflectionConfig
from agenix.knowledge.store import KnowledgeStore
from agenix.loader import load_agent
from agenix.storage.fs_backend import FSBackend
from agenix.storage.lineage import record_creation
from agenix.storage.models import (
    Difficulty,
    InsightCard,
    IterationResult,
    KnowledgeCard,
    LoadedAgent,
    Problem,
    ProblemStatus,
    ReflectionCard,
    ReflectionCategory,
    SourceReference,
    TestCase,
    TestResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


class AgentRunner(Protocol):
    """Protocol for invoking an agent and getting its text output."""

    def run(self, agent: LoadedAgent, input_payload: str) -> str: ...


class Pipeline:
    """Sequential pipeline that runs one iteration of the reflection loop."""

    def __init__(
        self,
        config: ReflectionConfig,
        runner: AgentRunner,
        knowledge_store: Optional[KnowledgeStore] = None,
        fs_backend: Optional[FSBackend] = None,
    ) -> None:
        self._config = config
        self._runner = runner
        self._fs = fs_backend or FSBackend(config.storage)
        self._store = knowledge_store or KnowledgeStore(
            config=config, fs_backend=self._fs
        )

    @property
    def fs(self) -> FSBackend:
        return self._fs

    @property
    def store(self) -> KnowledgeStore:
        return self._store

    def initialize(self) -> None:
        """Ensure storage directories exist."""
        self._fs.initialize()
        self._store.initialize()

    def run_iteration(
        self, run_tag: str, iteration: int = 1
    ) -> IterationResult:
        """Run a single iteration of the full pipeline.

        Steps:
        1. CURATOR generates a problem
        2. SOLVER attempts the problem
        3. CRITIC produces reflection cards
        4. ORGANIZER synthesizes knowledge cards
        5. (optional) INSIGHT_FINDER produces insight cards
        """
        self.initialize()

        # Step 1: Curator
        problem = self._run_curator(run_tag, iteration)
        logger.info("Curator produced problem: %s (%s)", problem.title, problem.problem_id)

        # Step 2: Solver
        trajectory = self._run_solver(run_tag, problem)
        logger.info(
            "Solver finished: correct=%s, trajectory=%s",
            trajectory.is_correct,
            trajectory.trajectory_id,
        )

        # Step 3: Critic
        reflection_cards = self._run_critic(run_tag, problem, trajectory)
        logger.info("Critic produced %d reflection cards", len(reflection_cards))

        # Step 4: Organizer
        knowledge_cards = self._run_organizer(run_tag, problem, trajectory, reflection_cards)
        logger.info("Organizer produced %d knowledge cards", len(knowledge_cards))

        # Step 5: Insight Finder (periodic)
        insight_cards: list[InsightCard] = []
        if self._should_run_insight_finder(iteration):
            insight_cards = self._run_insight_finder(run_tag, iteration)
            logger.info("Insight Finder produced %d insight cards", len(insight_cards))

        all_card_ids = (
            [c.card_id for c in reflection_cards]
            + [c.card_id for c in knowledge_cards]
            + [c.card_id for c in insight_cards]
        )

        return IterationResult(
            run_tag=run_tag,
            problem_id=problem.problem_id,
            trajectory_id=trajectory.trajectory_id,
            is_correct=trajectory.is_correct,
            cards_created=all_card_ids,
        )

    # --- Agent Steps ---

    def _run_curator(self, run_tag: str, iteration: int) -> Problem:
        agent = load_agent("curator")
        previous_domains = [
            p.domain for p in self._fs.list_problems(limit=50)
        ]
        previous_difficulties = [
            p.difficulty.value for p in self._fs.list_problems(limit=50)
        ]

        input_payload = json.dumps({
            "iteration": iteration,
            "previous_domains": previous_domains,
            "previous_difficulties": previous_difficulties,
            "knowledge_hints": [],
        })

        output = self._runner.run(agent, input_payload)
        problem = _parse_problem(output)
        self._fs.save_problem(problem)
        return problem

    def _run_solver(self, run_tag: str, problem: Problem) -> Trajectory:
        agent = load_agent("solver")

        # Retrieve relevant knowledge
        knowledge_hits = self._store.search(
            query=f"{problem.title} {problem.domain}",
            limit=self._config.embedder.top_k,
        )
        knowledge = [
            {"title": r["title"], "content": r["card"].content, "card_type": r["card_type"]}
            for r in knowledge_hits
        ]

        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "knowledge": knowledge,
            "previous_attempts": [],
        })

        self._fs.update_problem_status(problem.problem_id, ProblemStatus.SOLVING)
        output = self._runner.run(agent, input_payload)
        trajectory = _parse_trajectory(output, problem.problem_id)
        self._fs.save_trajectory(trajectory, run_tag)

        new_status = ProblemStatus.SOLVED if trajectory.is_correct else ProblemStatus.FAILED
        self._fs.update_problem_status(problem.problem_id, new_status)
        return trajectory

    def _run_critic(
        self,
        run_tag: str,
        problem: Problem,
        trajectory: Trajectory,
    ) -> list[ReflectionCard]:
        agent = load_agent("critic")

        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "trajectory": json.loads(trajectory.model_dump_json()),
        })

        output = self._runner.run(agent, input_payload)
        cards = _parse_reflection_cards(output, trajectory.trajectory_id)

        for card in cards:
            source_refs = [
                SourceReference(id=trajectory.trajectory_id, type="trajectory"),
            ]
            record_creation(card, source_refs, agent="critic", run_tag=run_tag)
            self._store.add_card(card)

        return cards

    def _run_organizer(
        self,
        run_tag: str,
        problem: Problem,
        trajectory: Trajectory,
        reflection_cards: list[ReflectionCard],
    ) -> list[KnowledgeCard]:
        agent = load_agent("organizer")

        input_payload = json.dumps({
            "trajectory": json.loads(trajectory.model_dump_json()),
            "reflection_cards": [
                json.loads(c.model_dump_json()) for c in reflection_cards
            ],
            "problem": json.loads(problem.model_dump_json()),
        })

        output = self._runner.run(agent, input_payload)
        cards = _parse_knowledge_actions(output)

        for card in cards:
            source_refs = [
                SourceReference(id=trajectory.trajectory_id, type="trajectory"),
            ] + [
                SourceReference(id=rc.card_id, type="reflection")
                for rc in reflection_cards
            ]
            record_creation(card, source_refs, agent="organizer", run_tag=run_tag)
            self._store.add_card(card)

        return cards

    def _run_insight_finder(
        self, run_tag: str, iteration: int
    ) -> list[InsightCard]:
        agent = load_agent("insight_finder")

        recent = self._fs.list_trajectories(limit=20)
        if not recent:
            return []

        trajectories_data = []
        for t in recent:
            problem = self._fs.get_problem(t.problem_id)
            trajectories_data.append({
                "problem": json.loads(problem.model_dump_json()) if problem else {},
                "trajectory": json.loads(t.model_dump_json()),
            })

        input_payload = json.dumps({
            "trajectories": trajectories_data,
            "batch_info": {"run_tags": [run_tag], "total_count": len(recent)},
        })

        output = self._runner.run(agent, input_payload)
        cards = _parse_insight_cards(output)

        for card in cards:
            source_refs = [
                SourceReference(id=t.trajectory_id, type="trajectory")
                for t in recent
            ]
            record_creation(card, source_refs, agent="insight_finder", run_tag=run_tag)
            self._store.add_card(card)

        return cards

    def _should_run_insight_finder(self, iteration: int) -> bool:
        cfg = self._config.pipeline.insight_finder
        return cfg.enabled and iteration % cfg.frequency == 0


# --- Output Parsers ---


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from agent output text.

    Handles both raw JSON and JSON wrapped in markdown code blocks.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from agent output: {text[:200]}")


def _parse_problem(output: str) -> Problem:
    """Parse curator output into a Problem."""
    data = _extract_json(output)
    test_cases = [
        TestCase(**tc) for tc in data.get("test_cases", [])
    ]
    difficulty = Difficulty(data.get("difficulty", "medium"))
    return Problem(
        title=data["title"],
        description=data["description"],
        test_cases=test_cases,
        domain=data.get("domain", "general"),
        difficulty=difficulty,
    )


def _parse_trajectory(output: str, problem_id: str) -> Trajectory:
    """Parse solver output into a Trajectory."""
    data = _extract_json(output)
    test_results = [
        TestResult(
            test_case=TestCase(**tr["test_case"]),
            passed=tr["passed"],
            actual_output=tr.get("actual_output", ""),
            error=tr.get("error", ""),
        )
        for tr in data.get("test_results", [])
    ]
    return Trajectory(
        problem_id=problem_id,
        code_solution=data.get("code_solution", ""),
        final_answer=data.get("final_answer", ""),
        is_correct=data.get("is_correct", False),
        test_results=test_results,
        completed_at=datetime.now(timezone.utc),
    )


def _parse_reflection_cards(
    output: str, trajectory_id: str
) -> list[ReflectionCard]:
    """Parse critic output into ReflectionCards."""
    data = _extract_json(output)
    cards = []
    for rc in data.get("reflection_cards", []):
        try:
            category = ReflectionCategory(rc.get("category", "general"))
        except ValueError:
            category = ReflectionCategory.GENERAL
        cards.append(ReflectionCard(
            title=rc["title"],
            content=rc["content"],
            trajectory_id=trajectory_id,
            category=category,
            confidence=rc.get("confidence", 0.5),
            tags=rc.get("tags", []),
            supporting_steps=rc.get("supporting_steps", []),
        ))
    return cards


def _parse_knowledge_actions(output: str) -> list[KnowledgeCard]:
    """Parse organizer output into KnowledgeCards (create actions only for now)."""
    data = _extract_json(output)
    cards = []
    for action in data.get("actions", []):
        if action.get("action") != "create":
            # Revise/merge are handled in future phases
            continue
        cards.append(KnowledgeCard(
            title=action["title"],
            content=action["content"],
            domain=action.get("domain", "general"),
            applicability=action.get("applicability", ""),
            limitations=action.get("limitations", ""),
            tags=action.get("tags", []),
            related_card_ids=action.get("related_card_ids", []),
        ))
    return cards


def _parse_insight_cards(output: str) -> list[InsightCard]:
    """Parse insight finder output into InsightCards."""
    data = _extract_json(output)
    cards = []
    for ic in data.get("insight_cards", []):
        cards.append(InsightCard(
            title=ic["title"],
            content=ic["content"],
            hypothesis=ic.get("hypothesis", ""),
            evidence_for=ic.get("evidence_for", []),
            evidence_against=ic.get("evidence_against", []),
            tags=ic.get("tags", []),
        ))
    return cards
