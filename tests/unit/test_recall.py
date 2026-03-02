"""Unit tests for the recall tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agenix.storage.models import Card, Experience, Problem
from tools.recall.baseline.logic import create_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result: dict) -> dict:
    """Parse MCP tool result dict -> JSON payload."""
    text = result["content"][0]["text"]
    return json.loads(text)


def _is_error(result: dict) -> bool:
    return result.get("is_error", False)


def _make_recall(fs_backend=None):
    """Create a recall tool with optional mock FSBackend."""
    if fs_backend is None:
        fs_backend = MagicMock()
        fs_backend.get_problem.return_value = None
        fs_backend.get_experience.return_value = None
        fs_backend.get_card.return_value = None
    sdk_tool = create_tool(fs_backend=fs_backend)
    return sdk_tool.handler, fs_backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRecallProblem:
    @pytest.mark.asyncio
    async def test_recall_problem(self):
        problem = Problem(
            title="Fused Softmax",
            description="Write a Triton kernel for fused softmax.",
            domain="triton_kernels",
        )
        fs = MagicMock()
        fs.get_problem.return_value = problem
        handler, _ = _make_recall(fs)

        result = await handler({"entity_type": "problem", "entity_id": problem.problem_id})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["entity_type"] == "problem"
        assert data["entity_id"] == problem.problem_id
        assert data["data"]["title"] == "Fused Softmax"
        assert data["data"]["domain"] == "triton_kernels"

    @pytest.mark.asyncio
    async def test_problem_not_found(self):
        handler, _ = _make_recall()
        result = await handler({"entity_type": "problem", "entity_id": "nonexistent"})
        assert _is_error(result)
        assert "not found" in result["content"][0]["text"]


class TestRecallExperience:
    @pytest.mark.asyncio
    async def test_recall_experience(self):
        exp = Experience(
            problem_id="prob_123",
            code_solution="def solve(): pass",
            final_answer="done",
            is_correct=True,
        )
        fs = MagicMock()
        fs.get_experience.return_value = exp
        handler, _ = _make_recall(fs)

        result = await handler({"entity_type": "experience", "entity_id": exp.experience_id})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["entity_type"] == "experience"
        assert data["data"]["problem_id"] == "prob_123"
        assert data["data"]["is_correct"] is True

    @pytest.mark.asyncio
    async def test_experience_not_found(self):
        handler, _ = _make_recall()
        result = await handler({"entity_type": "experience", "entity_id": "nonexistent"})
        assert _is_error(result)


class TestRecallCard:
    @pytest.mark.asyncio
    async def test_recall_card(self):
        card = Card(
            card_type="reflection",
            title="Good use of tiling",
            content="Tiling improved performance by 2x.",
            category="optimization",
        )
        fs = MagicMock()
        fs.get_card.return_value = card
        handler, _ = _make_recall(fs)

        result = await handler({"entity_type": "card", "entity_id": card.card_id})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["entity_type"] == "card"
        assert data["data"]["title"] == "Good use of tiling"
        assert data["data"]["card_type"] == "reflection"

    @pytest.mark.asyncio
    async def test_card_not_found(self):
        handler, _ = _make_recall()
        result = await handler({"entity_type": "card", "entity_id": "nonexistent"})
        assert _is_error(result)


class TestRecallValidation:
    @pytest.mark.asyncio
    async def test_missing_entity_type(self):
        handler, _ = _make_recall()
        result = await handler({"entity_id": "abc"})
        assert _is_error(result)
        assert "entity_type is required" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_entity_id(self):
        handler, _ = _make_recall()
        result = await handler({"entity_type": "problem"})
        assert _is_error(result)
        assert "entity_id is required" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_invalid_entity_type(self):
        handler, _ = _make_recall()
        result = await handler({"entity_type": "bogus", "entity_id": "abc"})
        assert _is_error(result)
        assert "must be one of" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_empty_args(self):
        handler, _ = _make_recall()
        result = await handler({})
        assert _is_error(result)
