"""Parallel solver handler — runs N solver instances per problem concurrently.

Each thread gets its own ClaudeRunner (with isolated ToolRegistry and MCP
servers) via a runner_factory callable.  Knowledge retrieval is done once
and shared across all N runs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from agenix.loader import load_agent
from agenix.queue.fs_queue import FSQueue
from agenix.queue.models import QueueMessage
from agenix.runner import ClaudeRunner
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import Problem, ProblemStatus
from tools.knowledge.baseline.store import KnowledgeStore

logger = logging.getLogger(__name__)


class ParallelSolverHandler:
    """Queue handler that fans out N parallel solver runs per problem."""

    def __init__(
        self,
        runner_factory: Callable[[], ClaudeRunner],
        fs_backend: FSBackend,
        knowledge_store: KnowledgeStore,
        experiences_queue: FSQueue,
        run_tag: str,
        *,
        parallel: int = 3,
        knowledge_limit: int = 10,
    ) -> None:
        self._runner_factory = runner_factory
        self._fs = fs_backend
        self._store = knowledge_store
        self._exp_queue = experiences_queue
        self._run_tag = run_tag
        self._parallel = parallel
        self._knowledge_limit = knowledge_limit

    def handle(self, message: QueueMessage) -> None:
        """Fan out N solver runs for one problem, collect experiences."""
        problem_id = message.payload["problem_id"]
        problem = self._fs.get_problem(problem_id)
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")

        logger.info(
            "Parallel solving (%d instances): %s (%s)",
            self._parallel, problem.title, problem_id,
        )

        # Retrieve knowledge once (shared across all N runs)
        knowledge = self._retrieve_knowledge(problem)

        # Build the input payload (same for all N runs)
        input_payload = json.dumps({
            "problem": json.loads(problem.model_dump_json()),
            "knowledge": knowledge,
            "previous_attempts": [],
        })

        # Run N solvers in parallel
        self._fs.update_problem_status(problem_id, ProblemStatus.SOLVING)
        experience_ids = self._run_parallel(problem_id, input_payload)

        if not experience_ids:
            self._fs.update_problem_status(problem_id, ProblemStatus.FAILED)
            raise RuntimeError(
                f"All {self._parallel} solver instances failed for {problem_id}"
            )

        self._fs.update_problem_status(problem_id, ProblemStatus.SOLVED)

        # Enqueue batch to experiences queue
        self._exp_queue.initialize()
        self._exp_queue.enqueue(
            sender="solver",
            payload={
                "problem_id": problem_id,
                "experience_ids": experience_ids,
            },
        )

        logger.info(
            "Parallel solver finished: %d/%d succeeded, experience_ids=%s",
            len(experience_ids), self._parallel, experience_ids,
        )

    def _retrieve_knowledge(self, problem: Problem) -> list[dict]:
        """Retrieve relevant knowledge cards for the problem."""
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
            if card.applicability:
                entry["applicability"] = card.applicability
            if card.limitations:
                entry["limitations"] = card.limitations
            knowledge.append(entry)
        return knowledge

    def _run_one(self, problem_id: str, input_payload: str, index: int) -> str | None:
        """Run a single solver instance. Returns experience_id or None."""
        try:
            runner = self._runner_factory()
            agent = load_agent("solver")
            logger.info("Solver instance %d starting for %s", index, problem_id)
            result = runner.run(agent, input_payload)

            if result.experience_id:
                logger.info(
                    "Solver instance %d succeeded: experience=%s",
                    index, result.experience_id,
                )
                return result.experience_id

            logger.warning("Solver instance %d produced no experience", index)
            return None
        except Exception:
            logger.exception("Solver instance %d failed", index)
            return None

    def _run_parallel(self, problem_id: str, input_payload: str) -> list[str]:
        """Run N solvers via ThreadPoolExecutor, return successful experience_ids."""
        with ThreadPoolExecutor(max_workers=self._parallel) as pool:
            futures = {
                pool.submit(self._run_one, problem_id, input_payload, i): i
                for i in range(self._parallel)
            }
            experience_ids: list[str] = []
            for future in as_completed(futures):
                eid = future.result()
                if eid:
                    experience_ids.append(eid)
        return experience_ids
