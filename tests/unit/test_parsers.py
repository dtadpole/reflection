"""Tests for the extracted parsers module."""

from __future__ import annotations

import json

import pytest

from agenix.parsers import (
    coerce_str,
    extract_json,
    parse_experience,
    parse_insight_cards,
    parse_knowledge_actions,
    parse_problem,
    parse_reflection_cards,
)
from agenix.storage.models import Difficulty, ReflectionCategory


class TestExtractJson:
    def test_raw_json(self):
        data = extract_json('{"key": "value"}')
        assert data == {"key": "value"}

    def test_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        data = extract_json(text)
        assert data == {"key": "value"}

    def test_prose_prefixed(self):
        text = 'Here is the result: {"key": "value"}'
        data = extract_json(text)
        assert data == {"key": "value"}

    def test_code_block_without_json_tag(self):
        text = '```\n{"key": "value"}\n```'
        data = extract_json(text)
        assert data == {"key": "value"}

    def test_whitespace_stripped(self):
        text = '  \n  {"key": "value"}  \n  '
        data = extract_json(text)
        assert data == {"key": "value"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            extract_json("not json at all")


class TestCoerceStr:
    def test_string_passthrough(self):
        assert coerce_str("hello") == "hello"

    def test_list_to_json(self):
        assert coerce_str([1, 2, 3]) == "[1, 2, 3]"

    def test_dict_to_json(self):
        result = coerce_str({"a": 1})
        assert json.loads(result) == {"a": 1}

    def test_int_to_json(self):
        assert coerce_str(42) == "42"

    def test_none_to_json(self):
        assert coerce_str(None) == "null"


class TestParseProblem:
    def test_basic(self):
        output = json.dumps({
            "title": "Test Problem",
            "description": "A test",
            "test_cases": [
                {"input": "1", "expected_output": "2", "description": "basic"}
            ],
            "domain": "triton_kernels",
            "difficulty": "easy",
        })
        problem = parse_problem(output)
        assert problem.title == "Test Problem"
        assert problem.domain == "triton_kernels"
        assert problem.difficulty == Difficulty.EASY
        assert len(problem.test_cases) == 1

    def test_non_string_test_values(self):
        output = json.dumps({
            "title": "Test",
            "description": "desc",
            "test_cases": [
                {"input": [1, 2], "expected_output": 3, "description": "non-string"}
            ],
            "difficulty": "medium",
        })
        problem = parse_problem(output)
        assert problem.test_cases[0].input == "[1, 2]"
        assert problem.test_cases[0].expected_output == "3"

    def test_missing_optional_fields(self):
        output = json.dumps({
            "title": "Minimal",
            "description": "desc",
        })
        problem = parse_problem(output)
        assert problem.domain == "general"
        assert problem.difficulty == Difficulty.MEDIUM
        assert len(problem.test_cases) == 0


class TestParseExperience:
    def test_basic(self):
        output = json.dumps({
            "code_solution": "def solve(): pass",
            "final_answer": "done",
            "is_correct": True,
            "test_results": [],
        })
        exp = parse_experience(output, "prob_123")
        assert exp.problem_id == "prob_123"
        assert exp.code_solution == "def solve(): pass"
        assert exp.is_correct is True
        assert exp.completed_at is not None

    def test_with_test_results(self):
        output = json.dumps({
            "code_solution": "code",
            "final_answer": "ans",
            "is_correct": False,
            "test_results": [
                {
                    "test_case": {"input": "1", "expected_output": "2"},
                    "passed": False,
                    "actual_output": "3",
                    "error": "wrong",
                }
            ],
        })
        exp = parse_experience(output, "p1")
        assert len(exp.test_results) == 1
        assert exp.test_results[0].passed is False


class TestParseReflectionCards:
    def test_basic(self):
        output = json.dumps({
            "reflection_cards": [
                {
                    "title": "Good pattern",
                    "content": "Used tiling effectively",
                    "category": "optimization",
                    "confidence": 0.9,
                    "tags": ["tiling"],
                    "supporting_steps": [0, 1],
                }
            ]
        })
        cards = parse_reflection_cards(output, "exp_1")
        assert len(cards) == 1
        assert cards[0].title == "Good pattern"
        assert cards[0].experience_id == "exp_1"
        assert cards[0].category == ReflectionCategory.OPTIMIZATION
        assert cards[0].confidence == 0.9

    def test_invalid_category_defaults_to_general(self):
        output = json.dumps({
            "reflection_cards": [
                {
                    "title": "Test",
                    "content": "content",
                    "category": "nonexistent_category",
                    "confidence": 0.5,
                    "tags": [],
                    "supporting_steps": [],
                }
            ]
        })
        cards = parse_reflection_cards(output, "exp_1")
        assert cards[0].category == ReflectionCategory.GENERAL


class TestParseKnowledgeActions:
    def test_create_action(self):
        output = json.dumps({
            "actions": [
                {
                    "action": "create",
                    "title": "Tiling Strategy",
                    "content": "Use block tiling for matmul",
                    "domain": "triton_kernels",
                    "applicability": "Matrix multiply operations",
                    "limitations": "Small matrices may not benefit",
                    "tags": ["tiling", "matmul"],
                    "related_card_ids": [],
                }
            ]
        })
        cards = parse_knowledge_actions(output)
        assert len(cards) == 1
        assert cards[0].title == "Tiling Strategy"
        assert cards[0].domain == "triton_kernels"

    def test_skips_non_create_actions(self):
        output = json.dumps({
            "actions": [
                {"action": "revise", "card_id": "abc", "title": "Updated"},
                {"action": "merge", "card_ids": ["a", "b"], "title": "Merged"},
                {
                    "action": "create",
                    "title": "New",
                    "content": "content",
                    "tags": [],
                },
            ]
        })
        cards = parse_knowledge_actions(output)
        assert len(cards) == 1
        assert cards[0].title == "New"


class TestParseInsightCards:
    def test_basic(self):
        output = json.dumps({
            "insight_cards": [
                {
                    "title": "FP16 precision issue",
                    "content": "Reduction kernels fail with FP16 accumulators",
                    "hypothesis": "FP16 accumulation causes > 80% failure rate",
                    "evidence_for": ["softmax failed", "layernorm failed"],
                    "evidence_against": [],
                    "tags": ["precision", "fp16"],
                }
            ]
        })
        cards = parse_insight_cards(output)
        assert len(cards) == 1
        assert cards[0].title == "FP16 precision issue"
        assert len(cards[0].evidence_for) == 2

    def test_empty_list(self):
        output = json.dumps({"insight_cards": []})
        cards = parse_insight_cards(output)
        assert len(cards) == 0
