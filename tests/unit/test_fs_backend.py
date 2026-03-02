"""Tests for the filesystem storage backend."""

from __future__ import annotations

import pytest

from agenix.config import StorageConfig
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import (
    Card,
    Difficulty,
    Experience,
    ExperienceStep,
    Problem,
    ProblemStatus,
    StepType,
    TestCase,
)


def _write_experience_log(
    backend: FSBackend, experience_id: str, problem_id: str, agent: str = "solver"
) -> None:
    """Test helper: write a minimal .jsonl experience log."""
    import json

    agent_dir = backend.experiences_dir(agent)
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / f"{experience_id}.jsonl"
    user_msg = json.dumps({
        "role": "user",
        "content": json.dumps({"problem": {"problem_id": problem_id}}),
    })
    path.write_text(user_msg + "\n")


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
def sample_experience(sample_problem):
    return Experience(
        problem_id=sample_problem.problem_id,
        steps=[
            ExperienceStep(
                step_index=0, step_type=StepType.THOUGHT, content="Think about it"
            ),
            ExperienceStep(
                step_index=1,
                step_type=StepType.ACTION,
                content="Write code",
                tool_name="verifier",
                tool_input='{"reference_code": "...", "generated_code": "..."}',
            ),
        ],
        code_solution="def fib(n): ...",
        is_correct=True,
    )


@pytest.fixture
def sample_knowledge_card():
    return Card(
        card_type="knowledge",
        title="Dynamic Programming Basics",
        content="DP is about breaking problems into overlapping subproblems.",
        tags=["dp", "algorithms"],
        domain="algorithms",
        applicability="Problems with optimal substructure",
    )


@pytest.fixture
def sample_insight_card():
    return Card(
        card_type="insight",
        title="Memoization vs Tabulation",
        content="Top-down memoization often uses less memory for sparse problems.",
        tags=["dp", "optimization"],
        hypothesis="Memoization is faster for sparse inputs",
        hypothesis_status="proposed",
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


class TestExperienceCRUD:
    def test_get_experience_log(self, backend, sample_problem):
        _write_experience_log(backend, "exp_001", sample_problem.problem_id)
        log = backend.get_experience_log("exp_001")
        assert log is not None
        assert "problem_id" in log

    def test_get_nonexistent(self, backend):
        assert backend.get_experience_log("nonexistent") is None

    def test_list_experience_ids(self, backend, sample_problem):
        _write_experience_log(backend, "exp_001", sample_problem.problem_id)
        _write_experience_log(backend, "exp_002", sample_problem.problem_id)
        ids = backend.list_experience_ids()
        assert len(ids) == 2
        assert "exp_001" in ids
        assert "exp_002" in ids

    def test_count(self, backend, sample_problem):
        _write_experience_log(backend, "exp_001", sample_problem.problem_id)
        _write_experience_log(backend, "exp_002", sample_problem.problem_id)
        assert backend.count_experiences() == 2


class TestReflectionCardCRUD:
    def test_save_and_get(self, backend, sample_experience):
        r = Card(
            card_type="reflection",
            title="Memoization Insight",
            content="Fibonacci can be solved with memoization",
            experience_ids=[sample_experience.experience_id],
            reflection_confidence=0.9,
            supporting_steps=[0, 1],
        )
        backend.save_card(r)
        loaded = backend.get_card(r.card_id)
        assert loaded is not None
        assert loaded.card_type == "reflection"
        assert loaded.content == "Fibonacci can be solved with memoization"
        assert loaded.experience_ids == [sample_experience.experience_id]
        assert loaded.reflection_confidence == 0.9

    def test_list_by_type(self, backend, sample_knowledge_card, sample_experience):
        r = Card(
            card_type="reflection",
            title="Reflection",
            content="Content",
            experience_ids=[sample_experience.experience_id],
        )
        backend.save_card(sample_knowledge_card)
        backend.save_card(r)
        reflections = backend.list_cards(card_type="reflection")
        assert len(reflections) == 1
        assert reflections[0].card_type == "reflection"

    def test_list_by_experience(self, backend, sample_experience):
        r1 = Card(
            card_type="reflection",
            title="R1",
            content="C1",
            experience_ids=[sample_experience.experience_id],
        )
        r2 = Card(
            card_type="reflection",
            title="R2",
            content="C2",
            experience_ids=["other-exp"],
        )
        backend.save_card(r1)
        backend.save_card(r2)
        found = backend.list_cards_by_experience(sample_experience.experience_id)
        assert len(found) == 1
        assert found[0].card_id == r1.card_id


class TestCardCRUD:
    def test_save_and_get_knowledge_card(self, backend, sample_knowledge_card):
        backend.save_card(sample_knowledge_card)
        loaded = backend.get_card(sample_knowledge_card.card_id)
        assert loaded is not None
        assert loaded.card_type == "knowledge"
        assert loaded.title == "Dynamic Programming Basics"
        assert loaded.domain == "algorithms"

    def test_save_and_get_insight_card(self, backend, sample_insight_card):
        backend.save_card(sample_insight_card)
        loaded = backend.get_card(sample_insight_card.card_id)
        assert loaded is not None
        assert loaded.card_type == "insight"
        assert loaded.hypothesis_status == "proposed"

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
        knowledge = backend.list_cards(card_type="knowledge")
        assert len(knowledge) == 1
        assert knowledge[0].card_type == "knowledge"

    def test_count(self, backend, sample_knowledge_card, sample_insight_card):
        assert backend.count_cards() == 0
        backend.save_card(sample_knowledge_card)
        backend.save_card(sample_insight_card)
        assert backend.count_cards() == 2
        assert backend.count_cards(card_type="knowledge") == 1
        assert backend.count_cards(card_type="insight") == 1


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

    def test_query_experiences(self, backend, sample_problem):
        _write_experience_log(backend, "exp_001", sample_problem.problem_id)
        # query_experiences uses DuckDB over JSON — .jsonl files are also readable
        results = backend.query_experiences()
        assert len(results) >= 1

    def test_query_empty_dir(self, backend):
        results = backend.query_problems()
        assert results == []
