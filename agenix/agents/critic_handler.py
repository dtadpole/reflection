"""Critic handler — dequeues experiences, runs critic agent, produces reflection cards."""

from __future__ import annotations

import json
import logging

from agenix.loader import load_agent
from agenix.parsers import parse_reflection_cards
from agenix.queue.fs_queue import FSQueue
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
        reflections_queue: FSQueue,
        *,
        max_cards: int = 3,
    ) -> None:
        self._runner = runner
        self._fs = fs_backend
        self._store = knowledge_store
        self._reflections_queue = reflections_queue
        self._max_cards = max_cards

    def handle(self, message: QueueMessage) -> None:
        """Process an experience message from the experiences queue."""
        experience_id = message.payload["experience_id"]
        problem_id = message.payload["problem_id"]

        problem = self._fs.get_problem(problem_id)
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")

        logger.info(
            "Critiquing experience %s for problem %s",
            experience_id,
            problem.title,
        )

        # Give the critic metadata — it reads the experience via outline + excerpt tools
        input_payload = json.dumps({
            "problem_title": problem.title,
            "problem_id": problem_id,
            "experience_id": experience_id,
        })

        agent = load_agent("critic")
        result = self._runner.run(agent, input_payload)
        cards = parse_reflection_cards(result.output, [experience_id])
        cards = cards[:self._max_cards]

        for card in cards:
            source_refs = [
                SourceReference(id=experience_id, type="experience"),
            ]
            record_creation(card, source_refs, agent="critic")
            self._store.add_card(card)
            self._reflections_queue.enqueue(
                sender="critic",
                payload={"card_id": card.card_id},
            )

        logger.info("Critic produced %d reflection cards", len(cards))
