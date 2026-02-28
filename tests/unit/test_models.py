"""Tests for Pydantic data models."""

from __future__ import annotations

from agenix.storage.models import (
    AgentConfig,
    CardType,
    Difficulty,
    HypothesisStatus,
    InsightCard,
    KnowledgeCard,
    LoadedAgent,
    Problem,
    ProblemStatus,
    StepType,
    TestCase,
    TestResult,
    Trajectory,
    TrajectoryStep,
    Understanding,
    UnderstandingCategory,
)


class TestTestCase:
    def test_create_basic(self):
        tc = TestCase(input="[1,2,3]", expected_output="6")
        assert tc.input == "[1,2,3]"
        assert tc.expected_output == "6"
        assert tc.description == ""

    def test_with_description(self):
        tc = TestCase(input="0", expected_output="0", description="Edge case")
        assert tc.description == "Edge case"

    def test_serialization_roundtrip(self):
        tc = TestCase(input="hello", expected_output="world", description="test")
        data = tc.model_dump()
        tc2 = TestCase.model_validate(data)
        assert tc == tc2


class TestProblem:
    def test_create_with_defaults(self):
        p = Problem(title="Two Sum", description="Find two numbers")
        assert p.title == "Two Sum"
        assert p.difficulty == Difficulty.MEDIUM
        assert p.status == ProblemStatus.PROPOSED
        assert p.domain == "general"
        assert p.test_cases == []
        assert p.problem_id  # auto-generated

    def test_create_full(self):
        p = Problem(
            title="Fibonacci",
            description="Return nth fib",
            test_cases=[
                TestCase(input="0", expected_output="0"),
                TestCase(input="10", expected_output="55"),
            ],
            domain="dynamic_programming",
            difficulty=Difficulty.EASY,
        )
        assert len(p.test_cases) == 2
        assert p.difficulty == Difficulty.EASY

    def test_json_roundtrip(self):
        p = Problem(
            title="Test",
            description="Desc",
            test_cases=[TestCase(input="1", expected_output="1")],
        )
        json_str = p.model_dump_json()
        p2 = Problem.model_validate_json(json_str)
        assert p2.title == p.title
        assert len(p2.test_cases) == 1

    def test_unique_ids(self):
        p1 = Problem(title="A", description="A")
        p2 = Problem(title="B", description="B")
        assert p1.problem_id != p2.problem_id


class TestTrajectory:
    def test_create_empty(self):
        t = Trajectory(problem_id="test-123")
        assert t.problem_id == "test-123"
        assert t.steps == []
        assert t.is_correct is False
        assert t.completed_at is None

    def test_with_steps(self):
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.THOUGHT,
                content="Let me think about this",
            ),
            TrajectoryStep(
                step_index=1,
                step_type=StepType.ACTION,
                content="Running code",
                tool_name="code_executor",
                tool_input='print("hello")',
                tool_output="hello",
            ),
            TrajectoryStep(
                step_index=2,
                step_type=StepType.OBSERVATION,
                content="Code executed successfully",
            ),
        ]
        t = Trajectory(problem_id="p1", steps=steps)
        assert len(t.steps) == 3
        assert t.steps[1].tool_name == "code_executor"

    def test_serialization_roundtrip(self):
        t = Trajectory(
            problem_id="p1",
            steps=[
                TrajectoryStep(
                    step_index=0,
                    step_type=StepType.THOUGHT,
                    content="thinking",
                )
            ],
            code_solution="def solve(): pass",
            is_correct=True,
        )
        data = t.model_dump()
        t2 = Trajectory.model_validate(data)
        assert t2.is_correct is True
        assert t2.code_solution == "def solve(): pass"


class TestTestResult:
    def test_passed(self):
        tr = TestResult(
            test_case=TestCase(input="1", expected_output="1"),
            passed=True,
            actual_output="1",
        )
        assert tr.passed is True

    def test_failed_with_error(self):
        tr = TestResult(
            test_case=TestCase(input="bad", expected_output="good"),
            passed=False,
            error="ValueError: invalid input",
        )
        assert tr.passed is False
        assert "ValueError" in tr.error


class TestUnderstanding:
    def test_create(self):
        u = Understanding(
            trajectory_id="t1",
            content="Dynamic programming works by breaking problems into subproblems",
            category=UnderstandingCategory.ALGORITHM,
            confidence=0.8,
            supporting_steps=[0, 2, 4],
        )
        assert u.category == UnderstandingCategory.ALGORITHM
        assert u.confidence == 0.8
        assert len(u.supporting_steps) == 3

    def test_confidence_bounds(self):
        import pytest

        with pytest.raises(Exception):
            Understanding(
                trajectory_id="t1",
                content="test",
                confidence=1.5,
            )


class TestKnowledgeCard:
    def test_create(self):
        kc = KnowledgeCard(
            title="Binary Search Pattern",
            content="Binary search works on sorted arrays...",
            tags=["search", "arrays"],
            domain="algorithms",
            applicability="Sorted collections, monotonic functions",
            limitations="Requires random access, sorted input",
        )
        assert kc.card_type == CardType.KNOWLEDGE
        assert kc.domain == "algorithms"
        assert len(kc.tags) == 2
        assert kc.version == 1

    def test_json_roundtrip(self):
        kc = KnowledgeCard(
            title="Test Card",
            content="Content",
            tags=["test"],
            related_card_ids=["card-1", "card-2"],
        )
        json_str = kc.model_dump_json()
        kc2 = KnowledgeCard.model_validate_json(json_str)
        assert kc2.title == kc.title
        assert kc2.related_card_ids == ["card-1", "card-2"]


class TestInsightCard:
    def test_create(self):
        ic = InsightCard(
            title="Memoization improves recursive solutions",
            content="Adding memoization to recursive solutions...",
            hypothesis="Memoization reduces time complexity from exponential to polynomial",
            hypothesis_status=HypothesisStatus.CONFIRMED,
            evidence_for=["fib_trajectory_1", "dp_trajectory_3"],
            evidence_against=[],
            experiments_run=3,
        )
        assert ic.card_type == CardType.INSIGHT
        assert ic.hypothesis_status == HypothesisStatus.CONFIRMED
        assert ic.experiments_run == 3

    def test_defaults(self):
        ic = InsightCard(
            title="Test",
            content="Content",
        )
        assert ic.hypothesis_status == HypothesisStatus.PROPOSED
        assert ic.experiments_run == 0


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model == "sonnet"
        assert cfg.temperature == 0.7
        assert cfg.max_turns == 10
        assert cfg.tools == []
        assert cfg.custom_tools == []

    def test_custom(self):
        cfg = AgentConfig(
            model="opus",
            temperature=0.3,
            max_turns=20,
            tools=["Read", "Grep"],
            custom_tools=["code_executor"],
        )
        assert cfg.model == "opus"
        assert cfg.max_turns == 20


class TestLoadedAgent:
    def test_create(self):
        agent = LoadedAgent(
            name="test",
            description="A test agent",
            system_prompt="You are a test agent",
            config=AgentConfig(model="haiku"),
        )
        assert agent.name == "test"
        assert agent.config.model == "haiku"
        assert agent.logic_module_path is None
