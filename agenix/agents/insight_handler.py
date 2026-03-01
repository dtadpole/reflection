"""Insight finder handler — periodic cross-cutting pattern detection."""

from __future__ import annotations

import json
import logging

from agenix.execution_log import ExecutionLogger, NullExecutionLogger
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

    Analyzes recent trajectories to detect cross-cutting meta-patterns
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
        execution_log: ExecutionLogger | None = None,
    ) -> None:
        self._runner = runner
        self._fs = fs_backend
        self._store = knowledge_store
        self._run_tag = run_tag
        self._recent_limit = recent_limit
        self._log = execution_log or NullExecutionLogger()

    def handle(self) -> None:
        """Run one insight finder cycle over recent trajectories."""
        recent = self._fs.list_trajectories(limit=self._recent_limit)
        if not recent:
            logger.info("Insight finder: no trajectories to process")
            return

        trajectories_data = []
        for t in recent:
            problem = self._fs.get_problem(t.problem_id)
            trajectories_data.append({
                "problem": json.loads(problem.model_dump_json()) if problem else {},
                "trajectory": json.loads(t.model_dump_json()),
            })

        input_payload = json.dumps({
            "trajectories": trajectories_data,
            "batch_info": {
                "run_tags": [self._run_tag],
                "total_count": len(recent),
            },
        })

        agent = load_agent("insight_finder")
        result = self._runner.run(agent, input_payload)
        cards = parse_insight_cards(result.output)
        self._log.output_parsed(
            parser="parse_insight_cards",
            success=True,
            entities=[f"card:{c.card_id}" for c in cards],
        )

        for card in cards:
            source_refs = [
                SourceReference(id=t.trajectory_id, type="trajectory")
                for t in recent
            ]
            record_creation(
                card, source_refs, agent="insight_finder", run_tag=self._run_tag
            )
            self._store.add_card(card)
            self._log.data_saved("insight_card", card.card_id)

        logger.info("Insight finder produced %d insight cards", len(cards))
