"""Solver handler — dequeues problems, runs solver agent, enqueues experiences.

The solver agent internally iterates: write Triton kernel → verify → read
feedback → revise, using the Verifier and Knowledge Retriever MCP tools.
"""

from __future__ import annotations

import json
import logging

from agenix.loader import load_agent
from agenix.parsers import parse_experience
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
        knowledge = [
            {
                "title": r["title"],
                "content": r["card"].content,
                "card_type": r["card_type"],
            }
            for r in knowledge_hits
        ]

        # Build solver input
        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "knowledge": knowledge,
            "previous_attempts": [],
        })

        # Run solver agent
        self._fs.update_problem_status(problem_id, ProblemStatus.SOLVING)
        agent = load_agent("solver")
        result = self._runner.run(agent, input_payload)

        # Parse experience
        experience = parse_experience(result.output, problem_id)

        self._fs.save_experience(experience)

        new_status = (
            ProblemStatus.SOLVED if experience.is_correct else ProblemStatus.FAILED
        )
        self._fs.update_problem_status(problem_id, new_status)

        # Enqueue for critic
        self._exp_queue.initialize()
        self._exp_queue.enqueue(
            sender="solver",
            payload={
                "experience_id": experience.experience_id,
                "problem_id": problem_id,
            },
        )

        logger.info(
            "Solver finished: correct=%s, experience=%s",
            experience.is_correct,
            experience.experience_id,
        )
