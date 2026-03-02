"""Tests for Pydantic data models."""

from __future__ import annotations

from agenix.storage.models import (
    AgentConfig,
    Card,
    CardStatus,
    Difficulty,
    Experience,
    ExperienceStep,
    LineageEvent,
    LineageOperation,
    LoadedAgent,
    Problem,
    ProblemStatus,
    SourceReference,
    StepType,
    TestCase,
    TestResult,
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


class TestExperience:
    def test_create_empty(self):
        e = Experience(problem_id="test-123")
        assert e.problem_id == "test-123"
        assert e.steps == []
        assert e.is_correct is False
        assert e.completed_at is None

    def test_with_steps(self):
        steps = [
            ExperienceStep(
                step_index=0,
                step_type=StepType.THOUGHT,
                content="Let me think about this",
            ),
            ExperienceStep(
                step_index=1,
                step_type=StepType.ACTION,
                content="Running code",
                tool_name="verifier",
                tool_input='{"reference_code": "...", "generated_code": "..."}',
                tool_output='{"compiled": true, "correctness": true}',
            ),
            ExperienceStep(
                step_index=2,
                step_type=StepType.OBSERVATION,
                content="Verification completed successfully",
            ),
        ]
        e = Experience(problem_id="p1", steps=steps)
        assert len(e.steps) == 3
        assert e.steps[1].tool_name == "verifier"

    def test_serialization_roundtrip(self):
        e = Experience(
            problem_id="p1",
            steps=[
                ExperienceStep(
                    step_index=0,
                    step_type=StepType.THOUGHT,
                    content="thinking",
                )
            ],
            code_solution="def solve(): pass",
            is_correct=True,
        )
        data = e.model_dump()
        e2 = Experience.model_validate(data)
        assert e2.is_correct is True
        assert e2.code_solution == "def solve(): pass"


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


class TestReflectionCard:
    def test_create(self):
        r = Card(
            card_type="reflection",
            title="DP Reflection",
            content="Dynamic programming works by breaking problems into subproblems",
            experience_ids=["e1"],
        )
        assert r.card_type == "reflection"
        assert r.experience_ids == ["e1"]
        assert r.card_id

    def test_card_fields(self):
        r = Card(
            card_type="reflection",
            title="Test",
            content="Content",
            experience_ids=["e1"],
            tags=["dp"],
        )
        assert r.status == CardStatus.ACTIVE
        assert r.lineage == []
        assert r.source_refs == []
        assert r.tags == ["dp"]

    def test_json_roundtrip(self):
        r = Card(
            card_type="reflection",
            title="Test",
            content="Content",
            experience_ids=["e1"],
        )
        json_str = r.model_dump_json()
        r2 = Card.model_validate_json(json_str)
        assert r2.card_type == "reflection"
        assert r2.experience_ids == ["e1"]


class TestKnowledgeCard:
    def test_create(self):
        kc = Card(
            card_type="knowledge",
            title="Binary Search Pattern",
            content="Binary search works on sorted arrays...",
            tags=["search", "arrays"],
            applicability="Sorted collections, monotonic functions",
            limitations="Requires random access, sorted input",
        )
        assert kc.card_type == "knowledge"
        assert len(kc.tags) == 2

    def test_json_roundtrip(self):
        kc = Card(
            card_type="knowledge",
            title="Test Card",
            content="Content",
            tags=["test"],
        )
        json_str = kc.model_dump_json()
        kc2 = Card.model_validate_json(json_str)
        assert kc2.title == kc.title


class TestInsightCard:
    def test_create(self):
        ic = Card(
            card_type="insight",
            title="Memoization improves recursive solutions",
            content="Adding memoization to recursive solutions...",
        )
        assert ic.card_type == "insight"

    def test_defaults(self):
        ic = Card(
            card_type="insight",
            title="Test",
            content="Content",
        )
        assert ic.card_type == "insight"


class TestCardStatus:
    def test_values(self):
        assert CardStatus.ACTIVE == "active"
        assert CardStatus.SUPERSEDED == "superseded"
        assert CardStatus.ARCHIVED == "archived"


class TestLineageOperation:
    def test_values(self):
        assert LineageOperation.CREATE == "create"
        assert LineageOperation.REVISE == "revise"
        assert LineageOperation.MERGE == "merge"
        assert LineageOperation.SPLIT == "split"
        assert LineageOperation.SUPERSEDE == "supersede"
        assert LineageOperation.ARCHIVE == "archive"


class TestSourceReference:
    def test_create(self):
        ref = SourceReference(id="traj-001", type="experience")
        assert ref.id == "traj-001"
        assert ref.type == "experience"

    def test_json_roundtrip(self):
        ref = SourceReference(id="refl-001", type="reflection")
        data = ref.model_dump()
        ref2 = SourceReference.model_validate(data)
        assert ref == ref2


class TestLineageEvent:
    def test_create_event(self):
        event = LineageEvent(
            operation=LineageOperation.CREATE,
            agent="organizer",
            source_refs=[SourceReference(id="traj-1", type="experience")],
        )
        assert event.operation == LineageOperation.CREATE
        assert event.agent == "organizer"
        assert len(event.source_refs) == 1

    def test_merge_event(self):
        event = LineageEvent(
            operation=LineageOperation.MERGE,
            description="Merged from card-a, card-b",
        )
        assert event.operation == LineageOperation.MERGE

    def test_defaults(self):
        event = LineageEvent(operation=LineageOperation.ARCHIVE)
        assert event.agent == ""
        assert event.source_refs == []
        assert event.description == ""


class TestCardLineageFields:
    def test_new_fields_have_defaults(self):
        card = Card(title="Test", content="Content")
        assert card.status == CardStatus.ACTIVE
        assert card.lineage == []
        assert card.source_refs == []

    def test_insight_card_has_lineage_fields(self):
        card = Card(card_type="insight", title="Test", content="Content")
        assert card.status == CardStatus.ACTIVE
        assert card.lineage == []
        assert card.source_refs == []

    def test_card_with_lineage_json_roundtrip(self):
        card = Card(
            card_type="knowledge",
            title="Test",
            content="Content",
            status=CardStatus.SUPERSEDED,
            source_refs=[SourceReference(id="traj-1", type="experience")],
            lineage=[
                LineageEvent(
                    operation=LineageOperation.CREATE,
                    agent="organizer",
                    source_refs=[SourceReference(id="traj-1", type="experience")],
                )
            ],
        )
        json_str = card.model_dump_json()
        restored = Card.model_validate_json(json_str)
        assert restored.status == CardStatus.SUPERSEDED
        assert len(restored.lineage) == 1
        assert restored.lineage[0].operation == LineageOperation.CREATE


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
            custom_tools=["verifier"],
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
