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

    Reads recent experience logs and reflection cards from the knowledge base,
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
        recent_ids = self._fs.list_experience_ids(limit=self._recent_limit)
        if not recent_ids:
            logger.info("Organizer: no experiences to process")
            return

        # Gather raw experience logs
        experience_logs = []
        for eid in recent_ids:
            log_text = self._fs.get_experience_log(eid)
            if log_text:
                experience_logs.append({
                    "experience_id": eid,
                    "conversation_log": log_text,
                })

        # Gather recent reflection cards
        reflection_cards = self._fs.list_cards(card_type="reflection", limit=50)
        reflection_data = [
            json.loads(c.model_dump_json()) for c in reflection_cards
        ]

        input_payload = json.dumps({
            "experiences": experience_logs,
            "reflection_cards": reflection_data,
        })

        agent = load_agent("organizer")
        result = self._runner.run(agent, input_payload)
        exp_ids = recent_ids[:3]
        cards = parse_knowledge_actions(result.output, experience_ids=exp_ids)

        for card in cards:
            source_refs = [
                SourceReference(id=eid, type="experience")
                for eid in recent_ids
            ]
            record_creation(
                card, source_refs, agent="organizer", run_tag=self._run_tag
            )
            self._store.add_card(card)

        logger.info("Organizer produced %d knowledge cards", len(cards))
