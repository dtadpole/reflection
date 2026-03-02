"""Insight finder handler — periodic cross-cutting pattern detection."""

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

    Analyzes recent experiences to detect cross-cutting meta-patterns
    and produces insight cards.
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
        recent = self._fs.list_experiences(limit=self._recent_limit)
        if not recent:
            logger.info("Insight finder: no experiences to process")
            return

        experiences_data = []
        for e in recent:
            problem = self._fs.get_problem(e.problem_id)
            experiences_data.append({
                "problem": json.loads(problem.model_dump_json()) if problem else {},
                "experience": json.loads(e.model_dump_json()),
            })

        input_payload = json.dumps({
            "experiences": experiences_data,
            "batch_info": {
                "total_count": len(recent),
            },
        })

        agent = load_agent("insight_finder")
        result = self._runner.run(agent, input_payload)
        cards = parse_insight_cards(result.output)

        for card in cards:
            source_refs = [
                SourceReference(id=e.experience_id, type="experience")
                for e in recent
            ]
            record_creation(
                card, source_refs, agent="insight_finder", run_tag=self._run_tag
            )
            self._store.add_card(card)

        logger.info("Insight finder produced %d insight cards", len(cards))
