"""Solver handler — dequeues problems, runs solver agent, enqueues experiences.

The solver agent internally iterates: write Triton kernel → verify → read
feedback → revise, using the Verifier and Knowledge Retriever MCP tools.
"""

from __future__ import annotations

import json
import logging

from agenix.loader import load_agent
from agenix.queue.fs_queue import FSQueue
from agenix.queue.models import QueueMessage
from agenix.runner import ClaudeRunner
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import ProblemStatus
from tools.knowledge.baseline.store import KnowledgeStore

logger = logging.getLogger(__name__)


class SolverHandler:
    """Queue handler for the solver agent."""

    def __init__(
        self,
        runner: ClaudeRunner,
        fs_backend: FSBackend,
        knowledge_store: KnowledgeStore,
        experiences_queue: FSQueue,
        run_tag: str,
        *,
        knowledge_limit: int = 10,
    ) -> None:
        self._runner = runner
        self._fs = fs_backend
        self._store = knowledge_store
        self._exp_queue = experiences_queue
        self._run_tag = run_tag
        self._knowledge_limit = knowledge_limit

    def handle(self, message: QueueMessage) -> None:
        """Process a problem message from the problems queue."""
        problem_id = message.payload["problem_id"]
        problem = self._fs.get_problem(problem_id)
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")

        logger.info("Solving problem: %s (%s)", problem.title, problem_id)

        # Retrieve relevant knowledge cards using full problem context
        retrieval_query = (
            f"{problem.title} {problem.description[:500]} "
            f"{problem.domain} triton GPU kernel optimization"
        )
        knowledge_hits = self._store.search(
            query=retrieval_query,
            limit=self._knowledge_limit,
        )
        knowledge = []
        for r in knowledge_hits:
            card = r["card"]
            entry: dict = {
                "title": r["title"],
                "content": card.content,
                "card_type": r["card_type"],
                "tags": card.tags,
            }
            if card.code_snippet:
                entry["code_snippet"] = card.code_snippet
            # KnowledgeCard-specific fields
            if hasattr(card, "applicability") and card.applicability:
                entry["applicability"] = card.applicability
            if hasattr(card, "limitations") and card.limitations:
                entry["limitations"] = card.limitations
            knowledge.append(entry)

        # Build solver input
        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "knowledge": knowledge,
            "previous_attempts": [],
        })

        # Run solver agent — conversation is streamed to experience JSONL by runner
        self._fs.update_problem_status(problem_id, ProblemStatus.SOLVING)
        agent = load_agent("solver")
        result = self._runner.run(agent, input_payload)

        # Update problem status based on result
        is_error = result.output == "" or result.experience_id is None
        new_status = ProblemStatus.FAILED if is_error else ProblemStatus.SOLVED
        self._fs.update_problem_status(problem_id, new_status)

        # Enqueue for critic
        if result.experience_id:
            self._exp_queue.initialize()
            self._exp_queue.enqueue(
                sender="solver",
                payload={
                    "experience_id": result.experience_id,
                    "problem_id": problem_id,
                },
            )

        logger.info(
            "Solver finished: experience=%s, status=%s",
            result.experience_id,
            new_status.value,
        )
