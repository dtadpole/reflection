"""Critic handler — dequeues experiences, runs critic agent, produces reflection cards."""

from __future__ import annotations

import json
import logging

from agenix.loader import load_agent
from agenix.queue.fs_queue import FSQueue
from agenix.queue.models import QueueMessage
from agenix.runner import ClaudeRunner
from agenix.storage.fs_backend import FSBackend

logger = logging.getLogger(__name__)


class CriticHandler:
    """Queue handler for the critic agent.

    The critic agent creates reflection cards directly via the
    knowledge_create tool. The handler just enqueues created card IDs
    to the reflections queue for downstream consumers.
    """

    def __init__(
        self,
        runner: ClaudeRunner,
        fs_backend: FSBackend,
        reflections_queue: FSQueue,
    ) -> None:
        self._runner = runner
        self._fs = fs_backend
        self._reflections_queue = reflections_queue

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

        # Give the critic metadata — it reads the experience via recall tools
        # and creates reflection cards via knowledge_create tool
        input_payload = json.dumps({
            "problem_title": problem.title,
            "problem_id": problem_id,
            "experience_id": experience_id,
        })

        agent = load_agent("critic")
        self._runner.run(agent, input_payload)

        # Cards were created by agent via knowledge_create tool calls.
        # Find them by experience_id and enqueue to reflections queue.
        cards = self._fs.list_cards_by_experience(experience_id)
        for card in cards:
            self._reflections_queue.enqueue(
                sender="critic",
                payload={"card_id": card.card_id},
            )

        logger.info("Critic created %d cards via tool calls", len(cards))
