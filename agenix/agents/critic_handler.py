"""Critic handler — dequeues experiences, runs critic agent, produces reflection cards.

Supports both single-experience and batch (multi-experience) payloads.
When the experiences queue message contains ``experience_ids`` (plural),
the batch critic variant is loaded for comparative analysis across all
trajectories.  A single ``experience_id`` falls back to the base variant.
"""

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
        """Process an experience message from the experiences queue.

        Payload formats:
        - Single:  ``{"experience_id": "...", "problem_id": "..."}``
        - Batch:   ``{"experience_ids": ["...", ...], "problem_id": "..."}``
        """
        problem_id = message.payload["problem_id"]

        # Support both single and batch payloads
        experience_ids: list[str] = message.payload.get("experience_ids")  # type: ignore[assignment]
        if experience_ids is None:
            experience_ids = [message.payload["experience_id"]]

        problem = self._fs.get_problem(problem_id)
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")

        is_batch = len(experience_ids) > 1
        logger.info(
            "Critiquing %d experience(s) for problem %s%s",
            len(experience_ids),
            problem.title,
            " (batch)" if is_batch else "",
        )

        input_payload = json.dumps({
            "problem_title": problem.title,
            "problem_id": problem_id,
            "experience_ids": experience_ids,
        })

        variant = "batch" if is_batch else "base"
        agent = load_agent("critic", variant=variant)
        self._runner.run(agent, input_payload)

        # Cards were created by agent via knowledge_create tool calls.
        # Find them across all experience_ids and enqueue to reflections queue.
        total_cards = 0
        for eid in experience_ids:
            cards = self._fs.list_cards_by_experience(eid)
            for card in cards:
                self._reflections_queue.enqueue(
                    sender="critic",
                    payload={"card_id": card.card_id},
                )
            total_cards += len(cards)

        logger.info("Critic created %d cards via tool calls", total_cards)
