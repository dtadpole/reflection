"""Insight finder handler — periodic meta-pattern detection across experiences."""

from __future__ import annotations

import json
import logging

from agenix.loader import load_agent
from agenix.parsers import parse_insight_cards
from agenix.runner import ClaudeRunner
from agenix.storage.fs_backend import FSBackend
from agenix.storage.lineage import record_creation
from agenix.storage.models import SourceReference
from tools.knowledge.baseline.store import KnowledgeStore

logger = logging.getLogger(__name__)


class InsightHandler:
    """Scheduled handler for the insight finder agent.

    Reads recent experience logs and produces insight cards that capture
    cross-cutting meta-patterns.
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
        """Run one insight finder cycle over recent experiences."""
        recent_ids = self._fs.list_experience_ids(limit=self._recent_limit)
        if not recent_ids:
            logger.info("Insight finder: no experiences to process")
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

        input_payload = json.dumps({
            "experiences": experience_logs,
            "batch_info": {
                "total_count": len(experience_logs),
            },
        })

        agent = load_agent("insight_finder")
        result = self._runner.run(agent, input_payload)
        cards = parse_insight_cards(result.output)

        for card in cards:
            source_refs = [
                SourceReference(id=eid, type="experience")
                for eid in recent_ids
            ]
            record_creation(
                card, source_refs, agent="insight_finder", run_tag=self._run_tag
            )
            self._store.add_card(card)

        logger.info("Insight finder produced %d insight cards", len(cards))
