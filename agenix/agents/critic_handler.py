"""Critic handler — dequeues trajectories, runs critic agent, produces reflection cards."""

from __future__ import annotations

import json
import logging

from agenix.execution_log import ExecutionLogger, NullExecutionLogger
from agenix.loader import load_agent
from agenix.parsers import parse_reflection_cards
from agenix.queue.models import QueueMessage
from agenix.runner import ClaudeRunner
from agenix.storage.fs_backend import FSBackend
from agenix.storage.lineage import record_creation
from agenix.storage.models import SourceReference
from tools.knowledge.baseline.store import KnowledgeStore

logger = logging.getLogger(__name__)


class CriticHandler:
    """Queue handler for the critic agent."""

    def __init__(
        self,
        runner: ClaudeRunner,
        fs_backend: FSBackend,
        knowledge_store: KnowledgeStore,
        execution_log: ExecutionLogger | None = None,
    ) -> None:
        self._runner = runner
        self._fs = fs_backend
        self._store = knowledge_store
        self._log = execution_log or NullExecutionLogger()

    def handle(self, message: QueueMessage) -> None:
        """Process a trajectory message from the trajectories queue."""
        trajectory_id = message.payload["trajectory_id"]
        problem_id = message.payload["problem_id"]
        run_tag = message.payload["run_tag"]

        problem = self._fs.get_problem(problem_id)
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")

        trajectory = self._fs.get_trajectory(trajectory_id, run_tag=run_tag)
        if trajectory is None:
            raise ValueError(f"Trajectory {trajectory_id} not found")

        logger.info(
            "Critiquing trajectory %s for problem %s",
            trajectory_id,
            problem.title,
        )

        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "trajectory": json.loads(trajectory.model_dump_json()),
        })

        agent = load_agent("critic")
        result = self._runner.run(agent, input_payload)
        cards = parse_reflection_cards(result.output, trajectory_id)
        self._log.output_parsed(
            parser="parse_reflection_cards",
            success=True,
            entities=[f"card:{c.card_id}" for c in cards],
        )

        for card in cards:
            source_refs = [
                SourceReference(id=trajectory_id, type="trajectory"),
            ]
            record_creation(card, source_refs, agent="critic", run_tag=run_tag)
            self._store.add_card(card)
            self._log.data_saved("reflection_card", card.card_id)

        logger.info("Critic produced %d reflection cards", len(cards))
