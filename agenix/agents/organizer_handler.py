"""Organizer handler — periodic knowledge synthesis from recent trajectories."""

from __future__ import annotations

import json
import logging

from agenix.loader import load_agent
from agenix.parsers import parse_knowledge_actions
from agenix.runner import ClaudeRunner
from agenix.storage.fs_backend import FSBackend
from agenix.storage.lineage import record_creation
from agenix.storage.models import CardType, SourceReference
from tools.knowledge.baseline.store import KnowledgeStore

logger = logging.getLogger(__name__)


class OrganizerHandler:
    """Scheduled handler for the organizer agent.

    Reads recent trajectories and reflection cards from the knowledge base,
    runs the organizer agent, and produces knowledge cards.
    """

    def __init__(
        self,
        runner: ClaudeRunner,
        fs_backend: FSBackend,
        knowledge_store: KnowledgeStore,
        run_tag: str,
        *,
        recent_limit: int = 20,
    ) -> None:
        self._runner = runner
        self._fs = fs_backend
        self._store = knowledge_store
        self._run_tag = run_tag
        self._recent_limit = recent_limit

    def handle(self) -> None:
        """Run one organizer cycle over recent trajectories."""
        recent = self._fs.list_trajectories(limit=self._recent_limit)
        if not recent:
            logger.info("Organizer: no trajectories to process")
            return

        # Gather trajectory + problem data
        trajectories_data = []
        for t in recent:
            problem = self._fs.get_problem(t.problem_id)
            trajectories_data.append({
                "problem": json.loads(problem.model_dump_json()) if problem else {},
                "trajectory": json.loads(t.model_dump_json()),
            })

        # Gather recent reflection cards
        reflection_cards = self._fs.list_cards(card_type=CardType.REFLECTION, limit=50)
        reflection_data = [
            json.loads(c.model_dump_json()) for c in reflection_cards
        ]

        input_payload = json.dumps({
            "trajectories": trajectories_data,
            "reflection_cards": reflection_data,
        })

        agent = load_agent("organizer")
        output = self._runner.run(agent, input_payload)
        cards = parse_knowledge_actions(output)

        for card in cards:
            source_refs = [
                SourceReference(id=t.trajectory_id, type="trajectory")
                for t in recent
            ]
            record_creation(
                card, source_refs, agent="organizer", run_tag=self._run_tag
            )
            self._store.add_card(card)

        logger.info("Organizer produced %d knowledge cards", len(cards))
