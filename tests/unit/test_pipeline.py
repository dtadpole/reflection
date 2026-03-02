"""Tests for the pipeline orchestrator."""

from __future__ import annotations

import json

import pytest

from agenix.config import ReflectionConfig, StorageConfig
from agenix.parsers import (
    extract_json,
    parse_experience,
    parse_insight_cards,
    parse_knowledge_actions,
    parse_problem,
    parse_reflection_cards,
)
from agenix.pipeline import Pipeline
from agenix.runner import AgentResult
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import (
    Difficulty,
    IterationResult,
    LoadedAgent,
    ReflectionCategory,
)

# --- Fake runner ---


class FakeRunner:
    """Fake AgentRunner that returns canned responses per agent name."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    def run(self, agent: LoadedAgent, input_payload: str) -> AgentResult:
        self.calls.append((agent.name, input_payload))
        output = self._responses.get(agent.name, "{}")
        return AgentResult(output=output)


# --- Canned JSON outputs ---

CURATOR_OUTPUT = json.dumps({
    "title": "Reverse a String",
    "description": "Write a function that reverses a string.",
    "test_cases": [
        {"input": "hello", "expected_output": "olleh", "description": "Basic case"},
        {"input": "", "expected_output": "", "description": "Empty string"},
    ],
    "domain": "strings",
    "difficulty": "easy",
})

SOLVER_OUTPUT = json.dumps({
    "code_solution": "def solve(s): return s[::-1]",
    "final_answer": "Slice with step -1",
    "is_correct": True,
    "test_results": [
        {
            "test_case": {"input": "hello", "expected_output": "olleh"},
            "passed": True,
            "actual_output": "olleh",
        },
    ],
})

CRITIC_OUTPUT = json.dumps({
    "reflection_cards": [
        {
            "title": "Python slicing for reversal",
            "content": "Using s[::-1] is idiomatic Python for string reversal.",
            "category": "pattern",
            "confidence": 0.9,
            "tags": ["python", "slicing"],
            "supporting_steps": [0],
        },
    ],
})

ORGANIZER_OUTPUT = json.dumps({
    "actions": [
        {
            "action": "create",
            "title": "String Reversal Techniques",
            "content": "Python slicing s[::-1] is the most concise way to reverse a string.",
            "domain": "strings",
            "applicability": "When reversing sequences in Python",
            "limitations": "Python-specific idiom",
            "tags": ["python", "strings", "reversal"],
            "related_card_ids": [],
        },
    ],
})

INSIGHT_FINDER_OUTPUT = json.dumps({
    "insight_cards": [
        {
            "title": "Python idioms improve conciseness",
            "content": "The solver consistently uses Pythonic idioms when available.",
            "hypothesis": "Using language idioms leads to more correct solutions",
            "evidence_for": ["String reversal used s[::-1] successfully"],
            "evidence_against": [],
            "tags": ["python", "idioms"],
        },
    ],
})


# --- Parser Tests ---


class TestExtractJson:
    def test_raw_json(self):
        data = extract_json('{"key": "value"}')
        assert data == {"key": "value"}

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        data = extract_json(text)
        assert data == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n```\n{"key": "value"}\n```\nDone.'
        data = extract_json(text)
        assert data == {"key": "value"}

    def test_prose_prefixed_json(self):
        text = 'Here is my answer:\n{"key": "value"}'
        data = extract_json(text)
        assert data == {"key": "value"}

    def test_prose_prefixed_nested_json(self):
        text = 'Sure, here is the output:\n\n{"title": "Foo", "items": [1, 2, 3]}'
        data = extract_json(text)
        assert data == {"title": "Foo", "items": [1, 2, 3]}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            extract_json("not json at all")


class TestParseProblem:
    def test_basic(self):
        problem = parse_problem(CURATOR_OUTPUT)
        assert problem.title == "Reverse a String"
        assert problem.domain == "strings"
        assert problem.difficulty == Difficulty.EASY
        assert len(problem.test_cases) == 2

    def test_defaults(self):
        minimal = json.dumps({"title": "Foo", "description": "Bar"})
        problem = parse_problem(minimal)
        assert problem.domain == "general"
        assert problem.difficulty == Difficulty.MEDIUM

    def test_non_string_test_case_values(self):
        """Test cases with non-string input/output should be coerced to JSON strings."""
        data = json.dumps({
            "title": "Add Two Numbers",
            "description": "Add two numbers.",
            "test_cases": [
                {"input": [0, 1], "expected_output": 2},
                {"input": {"a": 1}, "expected_output": [1, 2, 3]},
            ],
        })
        problem = parse_problem(data)
        assert len(problem.test_cases) == 2
        assert problem.test_cases[0].input == "[0, 1]"
        assert problem.test_cases[0].expected_output == "2"
        assert problem.test_cases[1].input == '{"a": 1}'
        assert problem.test_cases[1].expected_output == "[1, 2, 3]"


class TestParseExperience:
    def test_basic(self):
        exp = parse_experience(SOLVER_OUTPUT, "prob_123")
        assert exp.problem_id == "prob_123"
        assert exp.is_correct is True
        assert exp.code_solution == "def solve(s): return s[::-1]"
        assert len(exp.test_results) == 1
        assert exp.completed_at is not None


class TestParseReflectionCards:
    def test_basic(self):
        cards = parse_reflection_cards(CRITIC_OUTPUT, "exp_123")
        assert len(cards) == 1
        card = cards[0]
        assert card.title == "Python slicing for reversal"
        assert card.experience_id == "exp_123"
        assert card.category == ReflectionCategory.PATTERN
        assert card.confidence == 0.9
        assert card.supporting_steps == [0]

    def test_invalid_category_defaults(self):
        output = json.dumps({
            "reflection_cards": [{
                "title": "Test",
                "content": "Content",
                "category": "nonexistent",
            }],
        })
        cards = parse_reflection_cards(output, "exp_1")
        assert cards[0].category == ReflectionCategory.GENERAL


class TestParseKnowledgeActions:
    def test_create_action(self):
        cards = parse_knowledge_actions(ORGANIZER_OUTPUT)
        assert len(cards) == 1
        card = cards[0]
        assert card.title == "String Reversal Techniques"
        assert card.domain == "strings"
        assert "Python slicing" in card.content

    def test_skips_non_create(self):
        output = json.dumps({
            "actions": [
                {"action": "revise", "card_id": "abc", "title": "X", "content": "Y"},
                {"action": "create", "title": "Z", "content": "W"},
            ],
        })
        cards = parse_knowledge_actions(output)
        assert len(cards) == 1
        assert cards[0].title == "Z"


class TestParseInsightCards:
    def test_basic(self):
        cards = parse_insight_cards(INSIGHT_FINDER_OUTPUT)
        assert len(cards) == 1
        card = cards[0]
        assert card.title == "Python idioms improve conciseness"
        assert len(card.evidence_for) == 1


# --- Pipeline Integration Tests ---


@pytest.fixture
def pipeline_setup(tmp_path):
    """Create a Pipeline with fake runner and tmp storage."""
    config = ReflectionConfig(
        storage=StorageConfig(data_root=str(tmp_path), env="test"),
    )
    fs = FSBackend(config.storage)
    fs.initialize()

    runner = FakeRunner({
        "Curator": CURATOR_OUTPUT,
        "Solver": SOLVER_OUTPUT,
        "Critic": CRITIC_OUTPUT,
        "Organizer": ORGANIZER_OUTPUT,
        "Insight Finder": INSIGHT_FINDER_OUTPUT,
    })

    # Use the pipeline without KnowledgeStore to avoid requiring
    # sentence-transformers in unit tests.
    pipeline = Pipeline(config=config, runner=runner, fs_backend=fs)
    return pipeline, runner, fs


class TestPipelineCurator:
    def test_run_curator(self, pipeline_setup):
        pipeline, runner, fs = pipeline_setup
        problem = pipeline._run_curator("run_test", iteration=1)
        assert problem.title == "Reverse a String"
        assert problem.domain == "strings"
        # Problem was saved
        loaded = fs.get_problem(problem.problem_id)
        assert loaded is not None
        assert loaded.title == "Reverse a String"

    def test_curator_receives_input(self, pipeline_setup):
        pipeline, runner, fs = pipeline_setup
        pipeline._run_curator("run_test", iteration=3)
        assert len(runner.calls) == 1
        name, payload = runner.calls[0]
        assert name == "Curator"
        data = json.loads(payload)
        assert data["iteration"] == 3


class TestPipelineSolver:
    def test_run_solver(self, pipeline_setup):
        pipeline, runner, fs = pipeline_setup
        problem = parse_problem(CURATOR_OUTPUT)
        fs.save_problem(problem)

        exp = pipeline._run_solver("run_test", problem)
        assert exp.is_correct is True
        assert exp.code_solution == "def solve(s): return s[::-1]"

        # Problem status updated
        loaded_problem = fs.get_problem(problem.problem_id)
        assert loaded_problem.status.value == "solved"


class TestPipelineCritic:
    def test_run_critic(self, pipeline_setup):
        pipeline, runner, fs = pipeline_setup
        problem = parse_problem(CURATOR_OUTPUT)
        exp = parse_experience(SOLVER_OUTPUT, problem.problem_id)

        cards = pipeline._run_critic("run_test", problem, exp)
        assert len(cards) == 1
        assert cards[0].experience_id == exp.experience_id


class TestIterationResult:
    def test_model(self):
        result = IterationResult(
            run_tag="run_test",
            problem_id="p1",
            experience_id="e1",
            is_correct=True,
            cards_created=["c1", "c2"],
        )
        assert result.run_tag == "run_test"
        assert len(result.cards_created) == 2

    def test_json_roundtrip(self):
        result = IterationResult(
            run_tag="run_test",
            problem_id="p1",
            experience_id="e1",
        )
        data = result.model_dump_json()
        loaded = IterationResult.model_validate_json(data)
        assert loaded.run_tag == "run_test"
        assert loaded.is_correct is False
        assert loaded.cards_created == []


class TestShouldRunInsightFinder:
    def test_runs_at_frequency(self, pipeline_setup):
        pipeline, _, _ = pipeline_setup
        # Default frequency is 5
        assert pipeline._should_run_insight_finder(5) is True
        assert pipeline._should_run_insight_finder(10) is True

    def test_skips_between(self, pipeline_setup):
        pipeline, _, _ = pipeline_setup
        assert pipeline._should_run_insight_finder(1) is False
        assert pipeline._should_run_insight_finder(3) is False

    def test_disabled(self, tmp_path):
        config = ReflectionConfig(
            storage=StorageConfig(data_root=str(tmp_path), env="test"),
        )
        config.pipeline.insight_finder.enabled = False
        pipeline = Pipeline(config=config, runner=FakeRunner())
        assert pipeline._should_run_insight_finder(5) is False
