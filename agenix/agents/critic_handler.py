"""Critic handler — dequeues experiences, runs critic agent, produces reflection cards."""

from __future__ import annotations

import json
import logging

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
    ) -> None:
        self._runner = runner
        self._fs = fs_backend
        self._store = knowledge_store

    def handle(self, message: QueueMessage) -> None:
        """Process an experience message from the experiences queue."""
        experience_id = message.payload["experience_id"]
        problem_id = message.payload["problem_id"]

        problem = self._fs.get_problem(problem_id)
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")

        experience = self._fs.get_experience(experience_id)
        if experience is None:
            raise ValueError(f"Experience {experience_id} not found")

        logger.info(
            "Critiquing experience %s for problem %s",
            experience_id,
            problem.title,
        )

        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "experience": json.loads(experience.model_dump_json()),
        })

        agent = load_agent("critic")
        result = self._runner.run(agent, input_payload)
        cards = parse_reflection_cards(result.output, experience_id)

        for card in cards:
            source_refs = [
                SourceReference(id=experience_id, type="experience"),
            ]
            record_creation(card, source_refs, agent="critic")
            self._store.add_card(card)

        logger.info("Critic produced %d reflection cards", len(cards))
