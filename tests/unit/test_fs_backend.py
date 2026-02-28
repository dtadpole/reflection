"""Tests for the filesystem storage backend."""

from __future__ import annotations

import pytest

from agenix.config import StorageConfig
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import (
    CardType,
    Difficulty,
    HypothesisStatus,
    InsightCard,
    KnowledgeCard,
    Problem,
    ProblemStatus,
    ReflectionCard,
    ReflectionCategory,
    StepType,
    TestCase,
    Trajectory,
    TrajectoryStep,
)


@pytest.fixture
def backend(tmp_path):
    config = StorageConfig(data_root=str(tmp_path), env="test")
    b = FSBackend(config)
    b.initialize()
    return b


@pytest.fixture
def sample_problem():
    return Problem(
        title="Fibonacci",
        description="Write a function to compute the nth Fibonacci number.",
        test_cases=[TestCase(input="5", expected_output="5")],
        domain="algorithms",
        difficulty=Difficulty.EASY,
    )


@pytest.fixture
def sample_trajectory(sample_problem):
    return Trajectory(
        problem_id=sample_problem.problem_id,
        steps=[
            TrajectoryStep(
                step_index=0, step_type=StepType.THOUGHT, content="Think about it"
            ),
            TrajectoryStep(
                step_index=1,
                step_type=StepType.ACTION,
                content="Write code",
                tool_name="code_executor",
                tool_input="def fib(n): ...",
            ),
        ],
        code_solution="def fib(n): ...",
        is_correct=True,
    )


@pytest.fixture
def sample_knowledge_card():
    return KnowledgeCard(
        title="Dynamic Programming Basics",
        content="DP is about breaking problems into overlapping subproblems.",
        tags=["dp", "algorithms"],
        domain="algorithms",
        applicability="Problems with optimal substructure",
    )


@pytest.fixture
def sample_insight_card():
    return InsightCard(
        title="Memoization vs Tabulation",
        content="Top-down memoization often uses less memory for sparse problems.",
        tags=["dp", "optimization"],
        hypothesis="Memoization is faster for sparse inputs",
        hypothesis_status=HypothesisStatus.PROPOSED,
    )


class TestProblemCRUD:
    def test_save_and_get(self, backend, sample_problem):
        backend.save_problem(sample_problem)
        loaded = backend.get_problem(sample_problem.problem_id)
        assert loaded is not None
        assert loaded.title == "Fibonacci"
        assert loaded.domain == "algorithms"
        assert len(loaded.test_cases) == 1

    def test_get_nonexistent(self, backend):
        assert backend.get_problem("nonexistent") is None

    def test_list_problems(self, backend, sample_problem):
        backend.save_problem(sample_problem)
        p2 = Problem(title="Sorting", description="Sort an array", domain="algorithms")
        backend.save_problem(p2)
        problems = backend.list_problems()
        assert len(problems) == 2

    def test_list_filter_by_status(self, backend, sample_problem):
        backend.save_problem(sample_problem)
        p2 = Problem(
            title="Sorting",
            description="Sort an array",
            status=ProblemStatus.SOLVED,
        )
        backend.save_problem(p2)
        proposed = backend.list_problems(status=ProblemStatus.PROPOSED)
        assert len(proposed) == 1
        assert proposed[0].title == "Fibonacci"

    def test_list_filter_by_domain(self, backend, sample_problem):
        backend.save_problem(sample_problem)
        p2 = Problem(title="Parsing", description="Parse HTML", domain="web")
        backend.save_problem(p2)
        algo = backend.list_problems(domain="algorithms")
        assert len(algo) == 1

    def test_update_status(self, backend, sample_problem):
        backend.save_problem(sample_problem)
        backend.update_problem_status(
            sample_problem.problem_id, ProblemStatus.SOLVING
        )
        loaded = backend.get_problem(sample_problem.problem_id)
        assert loaded.status == ProblemStatus.SOLVING

    def test_count(self, backend, sample_problem):
        assert backend.count_problems() == 0
        backend.save_problem(sample_problem)
        assert backend.count_problems() == 1
        assert backend.count_problems(status=ProblemStatus.PROPOSED) == 1
        assert backend.count_problems(status=ProblemStatus.SOLVED) == 0


class TestTrajectoryCRUD:
    def test_save_and_get(self, backend, sample_trajectory):
        run_tag = "run_test"
        backend.save_trajectory(sample_trajectory, run_tag)
        loaded = backend.get_trajectory(
            sample_trajectory.trajectory_id, run_tag
        )
        assert loaded is not None
        assert len(loaded.steps) == 2
        assert loaded.is_correct is True

    def test_get_nonexistent(self, backend):
        assert backend.get_trajectory("nonexistent", "run_test") is None

    def test_list_trajectories(self, backend, sample_problem):
        run_tag = "run_test"
        t1 = Trajectory(problem_id=sample_problem.problem_id, is_correct=True)
        t2 = Trajectory(problem_id=sample_problem.problem_id, is_correct=False)
        backend.save_trajectory(t1, run_tag)
        backend.save_trajectory(t2, run_tag)

        all_t = backend.list_trajectories(run_tag=run_tag)
        assert len(all_t) == 2

        correct = backend.list_trajectories(run_tag=run_tag, is_correct=True)
        assert len(correct) == 1

    def test_list_across_runs(self, backend, sample_problem):
        t1 = Trajectory(problem_id=sample_problem.problem_id, is_correct=True)
        t2 = Trajectory(problem_id=sample_problem.problem_id, is_correct=False)
        backend.save_trajectory(t1, "run_20260228_100000")
        backend.save_trajectory(t2, "run_20260228_110000")

        all_t = backend.list_trajectories()
        assert len(all_t) == 2

    def test_count(self, backend, sample_problem):
        run_tag = "run_test"
        t1 = Trajectory(problem_id=sample_problem.problem_id, is_correct=True)
        t2 = Trajectory(problem_id=sample_problem.problem_id, is_correct=False)
        backend.save_trajectory(t1, run_tag)
        backend.save_trajectory(t2, run_tag)
        assert backend.count_trajectories() == 2
        assert backend.count_trajectories(is_correct=True) == 1


class TestReflectionCardCRUD:
    def test_save_and_get(self, backend, sample_trajectory):
        r = ReflectionCard(
            title="Memoization Insight",
            content="Fibonacci can be solved with memoization",
            trajectory_id=sample_trajectory.trajectory_id,
            category=ReflectionCategory.ALGORITHM,
            confidence=0.9,
            supporting_steps=[0, 1],
        )
        backend.save_card(r)
        loaded = backend.get_card(r.card_id)
        assert loaded is not None
        assert isinstance(loaded, ReflectionCard)
        assert loaded.content == "Fibonacci can be solved with memoization"
        assert loaded.trajectory_id == sample_trajectory.trajectory_id
        assert loaded.category == ReflectionCategory.ALGORITHM

    def test_list_by_type(self, backend, sample_knowledge_card, sample_trajectory):
        r = ReflectionCard(
            title="Reflection",
            content="Content",
            trajectory_id=sample_trajectory.trajectory_id,
        )
        backend.save_card(sample_knowledge_card)
        backend.save_card(r)
        reflections = backend.list_cards(card_type=CardType.REFLECTION)
        assert len(reflections) == 1
        assert isinstance(reflections[0], ReflectionCard)

    def test_list_by_trajectory(self, backend, sample_trajectory):
        r1 = ReflectionCard(
            title="R1",
            content="C1",
            trajectory_id=sample_trajectory.trajectory_id,
        )
        r2 = ReflectionCard(
            title="R2",
            content="C2",
            trajectory_id="other-traj",
        )
        backend.save_card(r1)
        backend.save_card(r2)
        found = backend.list_cards_by_trajectory(sample_trajectory.trajectory_id)
        assert len(found) == 1
        assert found[0].card_id == r1.card_id


class TestCardCRUD:
    def test_save_and_get_knowledge_card(self, backend, sample_knowledge_card):
        backend.save_card(sample_knowledge_card)
        loaded = backend.get_card(sample_knowledge_card.card_id)
        assert loaded is not None
        assert isinstance(loaded, KnowledgeCard)
        assert loaded.title == "Dynamic Programming Basics"
        assert loaded.domain == "algorithms"

    def test_save_and_get_insight_card(self, backend, sample_insight_card):
        backend.save_card(sample_insight_card)
        loaded = backend.get_card(sample_insight_card.card_id)
        assert loaded is not None
        assert isinstance(loaded, InsightCard)
        assert loaded.hypothesis_status == HypothesisStatus.PROPOSED

    def test_get_nonexistent(self, backend):
        assert backend.get_card("nonexistent") is None

    def test_list_cards(self, backend, sample_knowledge_card, sample_insight_card):
        backend.save_card(sample_knowledge_card)
        backend.save_card(sample_insight_card)
        all_cards = backend.list_cards()
        assert len(all_cards) == 2

    def test_list_filter_by_type(self, backend, sample_knowledge_card, sample_insight_card):
        backend.save_card(sample_knowledge_card)
        backend.save_card(sample_insight_card)
        knowledge = backend.list_cards(card_type=CardType.KNOWLEDGE)
        assert len(knowledge) == 1
        assert isinstance(knowledge[0], KnowledgeCard)

    def test_count(self, backend, sample_knowledge_card, sample_insight_card):
        assert backend.count_cards() == 0
        backend.save_card(sample_knowledge_card)
        backend.save_card(sample_insight_card)
        assert backend.count_cards() == 2
        assert backend.count_cards(card_type=CardType.KNOWLEDGE) == 1
        assert backend.count_cards(card_type=CardType.INSIGHT) == 1


class TestDuckDBQueries:
    def test_query_problems(self, backend, sample_problem):
        backend.save_problem(sample_problem)
        results = backend.query_problems()
        assert len(results) == 1
        assert results[0]["title"] == "Fibonacci"

    def test_query_problems_with_filter(self, backend, sample_problem):
        backend.save_problem(sample_problem)
        results = backend.query_problems(sql_where="domain = 'algorithms'")
        assert len(results) == 1
        results = backend.query_problems(sql_where="domain = 'web'")
        assert len(results) == 0

    def test_query_cards(self, backend, sample_knowledge_card):
        backend.save_card(sample_knowledge_card)
        results = backend.query_cards()
        assert len(results) == 1

    def test_query_trajectories(self, backend, sample_trajectory):
        backend.save_trajectory(sample_trajectory, "run_20260228_100000")
        results = backend.query_trajectories(run_tag="run_20260228_100000")
        assert len(results) == 1

    def test_query_empty_dir(self, backend):
        results = backend.query_problems()
        assert results == []
