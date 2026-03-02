"""Tests for ParallelSolverHandler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agenix.agents.parallel_solver_handler import ParallelSolverHandler
from agenix.queue.models import QueueMessage
from agenix.runner import AgentResult
from agenix.storage.models import Problem, ProblemStatus


@pytest.fixture()
def problem():
    return Problem(
        problem_id="prob_001",
        title="Softmax Kernel",
        description="Write a Triton softmax kernel",
        domain="triton",
    )


@pytest.fixture()
def fs_backend(problem):
    fs = MagicMock()
    fs.get_problem.return_value = problem
    return fs


@pytest.fixture()
def knowledge_store():
    store = MagicMock()
    store.search.return_value = []
    return store


@pytest.fixture()
def experiences_queue():
    q = MagicMock()
    return q


@pytest.fixture()
def message():
    return QueueMessage(
        message_id="msg_001",
        queue_name="problems",
        sender="curator",
        payload={"problem_id": "prob_001", "title": "Softmax Kernel"},
    )


def _make_runner(experience_id: str | None = None) -> MagicMock:
    runner = MagicMock()
    runner.run.return_value = AgentResult(
        output="done",
        experience_id=experience_id,
    )
    return runner


class TestParallelSolverHandler:
    def test_all_succeed(self, fs_backend, knowledge_store, experiences_queue, message):
        """All N solvers succeed — all experience_ids enqueued."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return _make_runner(f"exp_{call_count:03d}")

        handler = ParallelSolverHandler(
            runner_factory=factory,
            fs_backend=fs_backend,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=3,
        )

        handler.handle(message)

        # All 3 runners created
        assert call_count == 3

        # Problem status updated to SOLVED
        fs_backend.update_problem_status.assert_any_call("prob_001", ProblemStatus.SOLVING)
        fs_backend.update_problem_status.assert_any_call("prob_001", ProblemStatus.SOLVED)

        # Batch payload enqueued with all 3 experience_ids
        experiences_queue.enqueue.assert_called_once()
        payload = experiences_queue.enqueue.call_args.kwargs["payload"]
        assert payload["problem_id"] == "prob_001"
        assert len(payload["experience_ids"]) == 3

    def test_some_fail(self, fs_backend, knowledge_store, experiences_queue, message):
        """Some solvers fail — proceed with successful ones."""
        results = [
            AgentResult(output="done", experience_id="exp_001"),
            AgentResult(output="", experience_id=None),  # failed
            AgentResult(output="done", experience_id="exp_003"),
        ]
        idx = 0

        def factory():
            nonlocal idx
            runner = MagicMock()
            runner.run.return_value = results[idx]
            idx += 1
            return runner

        handler = ParallelSolverHandler(
            runner_factory=factory,
            fs_backend=fs_backend,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=3,
        )

        handler.handle(message)

        # Still marked SOLVED (at least 1 succeeded)
        fs_backend.update_problem_status.assert_any_call("prob_001", ProblemStatus.SOLVED)

        payload = experiences_queue.enqueue.call_args.kwargs["payload"]
        assert len(payload["experience_ids"]) == 2

    def test_all_fail(self, fs_backend, knowledge_store, experiences_queue, message):
        """All solvers fail — raises RuntimeError, marked FAILED."""

        def factory():
            return _make_runner(None)

        handler = ParallelSolverHandler(
            runner_factory=factory,
            fs_backend=fs_backend,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=3,
        )

        with pytest.raises(RuntimeError, match="All 3 solver instances failed"):
            handler.handle(message)

        fs_backend.update_problem_status.assert_any_call("prob_001", ProblemStatus.FAILED)
        experiences_queue.enqueue.assert_not_called()

    def test_problem_not_found(self, knowledge_store, experiences_queue, message):
        fs = MagicMock()
        fs.get_problem.return_value = None

        handler = ParallelSolverHandler(
            runner_factory=lambda: _make_runner(),
            fs_backend=fs,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=2,
        )

        with pytest.raises(ValueError, match="Problem prob_001 not found"):
            handler.handle(message)

    def test_knowledge_retrieval_shared(
        self, fs_backend, knowledge_store, experiences_queue, message,
    ):
        """Knowledge retrieval happens once, not N times."""

        def factory():
            return _make_runner("exp_x")

        handler = ParallelSolverHandler(
            runner_factory=factory,
            fs_backend=fs_backend,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=3,
        )

        handler.handle(message)

        # search called exactly once (shared across all N runs)
        knowledge_store.search.assert_called_once()

    def test_runner_exception_handled(
        self, fs_backend, knowledge_store, experiences_queue, message,
    ):
        """A runner that raises an exception is treated as a failed run."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            runner = MagicMock()
            if call_count == 2:
                runner.run.side_effect = RuntimeError("MCP crashed")
            else:
                runner.run.return_value = AgentResult(
                    output="done", experience_id=f"exp_{call_count:03d}",
                )
            return runner

        handler = ParallelSolverHandler(
            runner_factory=factory,
            fs_backend=fs_backend,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=3,
        )

        handler.handle(message)

        payload = experiences_queue.enqueue.call_args.kwargs["payload"]
        # 2 out of 3 succeeded
        assert len(payload["experience_ids"]) == 2


    def test_thread_name_set(self, fs_backend, knowledge_store, experiences_queue, message):
        """Each solver thread gets its name set to solver#N."""
        import threading

        captured_names: list[str] = []

        def factory():
            captured_names.append(threading.current_thread().name)
            return _make_runner("exp_x")

        handler = ParallelSolverHandler(
            runner_factory=factory,
            fs_backend=fs_backend,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=3,
        )

        handler.handle(message)

        assert len(captured_names) == 3
        assert set(captured_names) == {"solver#1", "solver#2", "solver#3"}

    def test_agent_name_not_mutated(
        self, fs_backend, knowledge_store, experiences_queue, message,
    ):
        """Runner.run() receives the original agent (name not mutated to solver#N)."""
        captured_agent_names: list[str] = []

        def factory():
            runner = MagicMock()
            runner.run.return_value = AgentResult(
                output="done", experience_id="exp_x",
            )

            original_run = runner.run

            def capturing_run(agent, payload, **kwargs):
                captured_agent_names.append(agent.name)
                return original_run(agent, payload, **kwargs)

            runner.run = capturing_run
            return runner

        handler = ParallelSolverHandler(
            runner_factory=factory,
            fs_backend=fs_backend,
            knowledge_store=knowledge_store,
            experiences_queue=experiences_queue,
            run_tag="test_run",
            parallel=2,
        )

        handler.handle(message)

        assert len(captured_agent_names) == 2
        # agent.name must never be mutated to include "#" (e.g. "solver#1")
        assert all("#" not in n for n in captured_agent_names)
        # All instances get the same (original) agent name
        assert len(set(captured_agent_names)) == 1


class TestCriticHandlerBatch:
    """Test CriticHandler with batch payloads."""

    def test_batch_payload_selects_batch_variant(self):
        """experience_ids (plural) in payload triggers batch variant."""
        from agenix.agents.critic_handler import CriticHandler

        runner = MagicMock()
        fs = MagicMock()
        fs.get_problem.return_value = Problem(
            problem_id="p1", title="Test", description="desc",
        )
        fs.list_cards_by_experience.return_value = []
        reflections_q = MagicMock()

        handler = CriticHandler(runner, fs, reflections_q)

        msg = QueueMessage(
            message_id="m1",
            queue_name="experiences",
            sender="solver",
            payload={"problem_id": "p1", "experience_ids": ["e1", "e2", "e3"]},
        )

        with patch("agenix.agents.critic_handler.load_agent") as mock_load:
            mock_load.return_value = MagicMock()
            handler.handle(msg)
            mock_load.assert_called_once_with("critic", variant="batch")

    def test_single_payload_selects_base_variant(self):
        """Single experience_id payload triggers base variant."""
        from agenix.agents.critic_handler import CriticHandler

        runner = MagicMock()
        fs = MagicMock()
        fs.get_problem.return_value = Problem(
            problem_id="p1", title="Test", description="desc",
        )
        fs.list_cards_by_experience.return_value = []
        reflections_q = MagicMock()

        handler = CriticHandler(runner, fs, reflections_q)

        msg = QueueMessage(
            message_id="m1",
            queue_name="experiences",
            sender="solver",
            payload={"problem_id": "p1", "experience_id": "e1"},
        )

        with patch("agenix.agents.critic_handler.load_agent") as mock_load:
            mock_load.return_value = MagicMock()
            handler.handle(msg)
            mock_load.assert_called_once_with("critic", variant="base")

    def test_batch_collects_cards_across_experiences(self):
        """Cards from all experiences are enqueued to reflections."""
        from agenix.agents.critic_handler import CriticHandler
        from agenix.storage.models import Card

        runner = MagicMock()
        fs = MagicMock()
        fs.get_problem.return_value = Problem(
            problem_id="p1", title="Test", description="desc",
        )

        card_a = Card(card_id="c1", title="Card A", content="...")
        card_b = Card(card_id="c2", title="Card B", content="...")
        card_c = Card(card_id="c3", title="Card C", content="...")

        # e1 has 2 cards, e2 has 1 card
        fs.list_cards_by_experience.side_effect = [
            [card_a, card_b],
            [card_c],
        ]

        reflections_q = MagicMock()
        handler = CriticHandler(runner, fs, reflections_q)

        msg = QueueMessage(
            message_id="m1",
            queue_name="experiences",
            sender="solver",
            payload={"problem_id": "p1", "experience_ids": ["e1", "e2"]},
        )

        with patch("agenix.agents.critic_handler.load_agent") as mock_load:
            mock_load.return_value = MagicMock()
            handler.handle(msg)

        # 3 cards enqueued total
        assert reflections_q.enqueue.call_count == 3
        enqueued_ids = [
            call.kwargs["payload"]["card_id"]
            for call in reflections_q.enqueue.call_args_list
        ]
        assert set(enqueued_ids) == {"c1", "c2", "c3"}
