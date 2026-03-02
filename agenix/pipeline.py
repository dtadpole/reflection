"""Pipeline orchestrator for the reflection loop.

Runs agents in sequence: Curator -> Solver -> Critic -> Organizer,
with periodic Insight Finder runs. Each step builds input from
stored data, invokes the agent, and persists results.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Protocol

from agenix.config import ReflectionConfig
from agenix.loader import load_agent
from agenix.parsers import (
    parse_experience,
    parse_insight_cards,
    parse_knowledge_actions,
    parse_problem,
    parse_reflection_cards,
)
from agenix.storage.fs_backend import FSBackend
from agenix.storage.lineage import record_creation
from agenix.storage.models import (
    Experience,
    InsightCard,
    IterationResult,
    KnowledgeCard,
    LoadedAgent,
    Problem,
    ProblemStatus,
    ReflectionCard,
    SourceReference,
)
from tools.knowledge.baseline.store import KnowledgeStore

logger = logging.getLogger(__name__)


class AgentRunner(Protocol):
    """Protocol for invoking an agent and returning structured results."""

    def run(self, agent: LoadedAgent, input_payload: str) -> Any: ...


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
        experience = self._run_solver(run_tag, problem)
        logger.info(
            "Solver finished: correct=%s, experience=%s",
            experience.is_correct,
            experience.experience_id,
        )

        # Step 3: Critic
        reflection_cards = self._run_critic(run_tag, problem, experience)
        logger.info("Critic produced %d reflection cards", len(reflection_cards))

        # Step 4: Organizer
        knowledge_cards = self._run_organizer(run_tag, problem, experience, reflection_cards)
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
            experience_id=experience.experience_id,
            is_correct=experience.is_correct,
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

        result = self._runner.run(agent, input_payload)
        problem = parse_problem(result.output)
        self._fs.save_problem(problem)
        return problem

    def _run_solver(self, run_tag: str, problem: Problem) -> Experience:
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
        result = self._runner.run(agent, input_payload)
        experience = parse_experience(result.output, problem.problem_id)

        new_status = ProblemStatus.SOLVED if experience.is_correct else ProblemStatus.FAILED
        self._fs.update_problem_status(problem.problem_id, new_status)
        return experience

    def _run_critic(
        self,
        run_tag: str,
        problem: Problem,
        experience: Experience,
    ) -> list[ReflectionCard]:
        agent = load_agent("critic")

        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "experience": json.loads(experience.model_dump_json()),
        })

        result = self._runner.run(agent, input_payload)
        cards = parse_reflection_cards(result.output, [experience.experience_id])

        for card in cards:
            source_refs = [
                SourceReference(id=experience.experience_id, type="experience"),
            ]
            record_creation(card, source_refs, agent="critic", run_tag=run_tag)
            self._store.add_card(card)

        return cards

    def _run_organizer(
        self,
        run_tag: str,
        problem: Problem,
        experience: Experience,
        reflection_cards: list[ReflectionCard],
    ) -> list[KnowledgeCard]:
        agent = load_agent("organizer")

        input_payload = json.dumps({
            "experience": json.loads(experience.model_dump_json()),
            "reflection_cards": [
                json.loads(c.model_dump_json()) for c in reflection_cards
            ],
            "problem": json.loads(problem.model_dump_json()),
        })

        result = self._runner.run(agent, input_payload)
        cards = parse_knowledge_actions(result.output)

        for card in cards:
            source_refs = [
                SourceReference(id=experience.experience_id, type="experience"),
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

        recent = self._fs.list_experiences(limit=20)
        if not recent:
            return []

        experiences_data = []
        for e in recent:
            problem = self._fs.get_problem(e.problem_id)
            experiences_data.append({
                "problem": json.loads(problem.model_dump_json()) if problem else {},
                "experience": json.loads(e.model_dump_json()),
            })

        input_payload = json.dumps({
            "experiences": experiences_data,
            "batch_info": {"total_count": len(recent)},
        })

        result = self._runner.run(agent, input_payload)
        cards = parse_insight_cards(result.output)

        for card in cards:
            source_refs = [
                SourceReference(id=e.experience_id, type="experience")
                for e in recent
            ]
            record_creation(card, source_refs, agent="insight_finder", run_tag=run_tag)
            self._store.add_card(card)

        return cards

    def _should_run_insight_finder(self, iteration: int) -> bool:
        cfg = self._config.pipeline.insight_finder
        return cfg.enabled and iteration % cfg.frequency == 0
