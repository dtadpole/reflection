"""Organizer handler — periodic knowledge synthesis from recent experiences."""

from __future__ import annotations

import json
import logging

from agenix.loader import load_agent
from agenix.parsers import parse_knowledge_actions
from agenix.runner import ClaudeRunner
from agenix.storage.fs_backend import FSBackend
from agenix.storage.lineage import record_creation
from agenix.storage.models import SourceReference
from tools.knowledge.baseline.store import KnowledgeStore

logger = logging.getLogger(__name__)


class OrganizerHandler:
    """Scheduled handler for the organizer agent.

    Reads recent experiences and reflection cards from the knowledge base,
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
        """Run one organizer cycle over recent experiences."""
        recent = self._fs.list_experiences(limit=self._recent_limit)
        if not recent:
            logger.info("Organizer: no experiences to process")
            return

        # Gather experience + problem data
        experiences_data = []
        for e in recent:
            problem = self._fs.get_problem(e.problem_id)
            experiences_data.append({
                "problem": json.loads(problem.model_dump_json()) if problem else {},
                "experience": json.loads(e.model_dump_json()),
            })

        # Gather recent reflection cards
        reflection_cards = self._fs.list_cards(card_type="reflection", limit=50)
        reflection_data = [
            json.loads(c.model_dump_json()) for c in reflection_cards
        ]

        input_payload = json.dumps({
            "experiences": experiences_data,
            "reflection_cards": reflection_data,
        })

        agent = load_agent("organizer")
        result = self._runner.run(agent, input_payload)
        exp_ids = [e.experience_id for e in recent[:3]]
        cards = parse_knowledge_actions(result.output, experience_ids=exp_ids)

        for card in cards:
            source_refs = [
                SourceReference(id=e.experience_id, type="experience")
                for e in recent
            ]
            record_creation(
                card, source_refs, agent="organizer", run_tag=self._run_tag
            )
            self._store.add_card(card)

        logger.info("Organizer produced %d knowledge cards", len(cards))
