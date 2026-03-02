"""Unit tests for the recall MCP tools (fetch, outline, excerpt)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agenix.storage.models import Card, Problem
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


def _make_tools(fs_backend=None):
    """Create recall tools with optional mock FSBackend."""
    if fs_backend is None:
        fs_backend = MagicMock()
        fs_backend.get_problem.return_value = None
        fs_backend.get_experience_log.return_value = None
        fs_backend.get_card.return_value = None
    tools = create_tool(fs_backend=fs_backend)
    by_name = {t.name: t for t in tools}
    return by_name, fs_backend


def _get_handler(name: str, fs_backend=None):
    """Get the handler for a specific tool name."""
    by_name, fs = _make_tools(fs_backend)
    return by_name[name].handler, fs


def _make_log(n_rows: int = 10) -> str:
    lines = []
    for i in range(n_rows):
        lines.append(json.dumps({"role": "user" if i % 2 == 0 else "assistant", "row": i + 1}))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# fetch tests
# ---------------------------------------------------------------------------


class TestFetchProblem:
    @pytest.mark.asyncio
    async def test_fetch_problem(self):
        problem = Problem(
            title="Fused Softmax",
            description="Write a Triton kernel for fused softmax.",
            domain="triton_kernels",
        )
        fs = MagicMock()
        fs.get_problem.return_value = problem
        handler, _ = _get_handler("recall_fetch", fs)

        result = await handler({"entity_type": "problem", "entity_id": problem.problem_id})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["entity_type"] == "problem"
        assert data["format"] == "json"
        assert data["data"]["title"] == "Fused Softmax"

    @pytest.mark.asyncio
    async def test_problem_not_found(self):
        handler, _ = _get_handler("recall_fetch")
        result = await handler({"entity_type": "problem", "entity_id": "nonexistent"})
        assert _is_error(result)
        assert "not found" in result["content"][0]["text"]


class TestFetchExperience:
    @pytest.mark.asyncio
    async def test_fetch_experience(self):
        log_text = '{"role": "user", "content": "hello"}\n'
        fs = MagicMock()
        fs.get_experience_log.return_value = log_text
        handler, _ = _get_handler("recall_fetch", fs)

        result = await handler({"entity_type": "experience", "entity_id": "exp_001"})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["entity_type"] == "experience"
        assert data["format"] == "jsonl"
        assert "hello" in data["data"]

    @pytest.mark.asyncio
    async def test_experience_not_found(self):
        handler, _ = _get_handler("recall_fetch")
        result = await handler({"entity_type": "experience", "entity_id": "nonexistent"})
        assert _is_error(result)


class TestFetchCard:
    @pytest.mark.asyncio
    async def test_fetch_card(self):
        card = Card(
            card_type="reflection",
            title="Good use of tiling",
            content="Tiling improved performance by 2x.",
            category="optimization",
        )
        fs = MagicMock()
        fs.get_card.return_value = card
        handler, _ = _get_handler("recall_fetch", fs)

        result = await handler({"entity_type": "card", "entity_id": card.card_id})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["entity_type"] == "card"
        assert data["format"] == "json"
        assert data["data"]["title"] == "Good use of tiling"

    @pytest.mark.asyncio
    async def test_card_not_found(self):
        handler, _ = _get_handler("recall_fetch")
        result = await handler({"entity_type": "card", "entity_id": "nonexistent"})
        assert _is_error(result)


class TestFetchValidation:
    @pytest.mark.asyncio
    async def test_missing_entity_type(self):
        handler, _ = _get_handler("recall_fetch")
        result = await handler({"entity_id": "abc"})
        assert _is_error(result)
        assert "entity_type is required" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_entity_id(self):
        handler, _ = _get_handler("recall_fetch")
        result = await handler({"entity_type": "problem"})
        assert _is_error(result)
        assert "entity_id is required" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_invalid_entity_type(self):
        handler, _ = _get_handler("recall_fetch")
        result = await handler({"entity_type": "bogus", "entity_id": "abc"})
        assert _is_error(result)
        assert "must be one of" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_empty_args(self):
        handler, _ = _get_handler("recall_fetch")
        result = await handler({})
        assert _is_error(result)


# ---------------------------------------------------------------------------
# outline tests
# ---------------------------------------------------------------------------


class TestOutlineExperience:
    @pytest.mark.asyncio
    async def test_outline_jsonl(self):
        fs = MagicMock()
        fs.get_experience_log.return_value = _make_log(5)
        handler, _ = _get_handler("recall_outline", fs)

        result = await handler({"entity_type": "experience", "entity_id": "exp_001"})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["format"] == "jsonl"
        assert data["total_messages"] == 5
        assert len(data["messages"]) == 5
        assert data["messages"][0]["row"] == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"
        assert all("length" in m for m in data["messages"])
        assert data["total_length"] > 0

    @pytest.mark.asyncio
    async def test_outline_not_found(self):
        handler, _ = _get_handler("recall_outline")
        result = await handler({"entity_type": "experience", "entity_id": "nonexistent"})
        assert _is_error(result)
        assert "not found" in result["content"][0]["text"]


class TestOutlineProblem:
    @pytest.mark.asyncio
    async def test_outline_json(self):
        problem = Problem(
            title="Fused Softmax",
            description="Write a Triton kernel.",
            domain="triton_kernels",
        )
        fs = MagicMock()
        fs.get_problem.return_value = problem
        handler, _ = _get_handler("recall_outline", fs)

        result = await handler({"entity_type": "problem", "entity_id": problem.problem_id})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["format"] == "json"
        assert data["length"] > 0
        assert "messages" not in data


class TestOutlineCard:
    @pytest.mark.asyncio
    async def test_outline_card_json(self):
        card = Card(
            card_type="reflection",
            title="Tiling",
            content="Tiling is good.",
            category="optimization",
        )
        fs = MagicMock()
        fs.get_card.return_value = card
        handler, _ = _get_handler("recall_outline", fs)

        result = await handler({"entity_type": "card", "entity_id": card.card_id})
        assert not _is_error(result)

        data = _parse_result(result)
        assert data["format"] == "json"
        assert data["length"] > 0


class TestOutlineValidation:
    @pytest.mark.asyncio
    async def test_missing_entity_type(self):
        handler, _ = _get_handler("recall_outline")
        result = await handler({"entity_id": "abc"})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_invalid_entity_type(self):
        handler, _ = _get_handler("recall_outline")
        result = await handler({"entity_type": "bogus", "entity_id": "abc"})
        assert _is_error(result)


# ---------------------------------------------------------------------------
# excerpt tests
# ---------------------------------------------------------------------------


def _make_excerpt_handler(log_text: str | None = None):
    fs = MagicMock()
    fs.get_experience_log.return_value = log_text
    handler, _ = _get_handler("recall_excerpt", fs)
    return handler


class TestExcerptBasic:
    @pytest.mark.asyncio
    async def test_read_all_rows(self):
        handler = _make_excerpt_handler(_make_log(5))
        result = await handler({"experience_id": "exp_001"})
        assert not _is_error(result)
        data = _parse_result(result)
        assert data["total_rows"] == 5
        assert len(data["rows"]) == 5
        assert data["start_row"] == 1
        assert data["end_row"] == 5

    @pytest.mark.asyncio
    async def test_read_range(self):
        handler = _make_excerpt_handler(_make_log(10))
        result = await handler({"experience_id": "exp_001", "start_row": 3, "end_row": 7})
        assert not _is_error(result)
        data = _parse_result(result)
        assert len(data["rows"]) == 5
        assert data["start_row"] == 3
        assert data["end_row"] == 7
        assert data["rows"][0]["row"] == 3

    @pytest.mark.asyncio
    async def test_clamp_end_row(self):
        handler = _make_excerpt_handler(_make_log(5))
        result = await handler({"experience_id": "exp_001", "start_row": 3, "end_row": 100})
        data = _parse_result(result)
        assert data["end_row"] == 5
        assert len(data["rows"]) == 3

    @pytest.mark.asyncio
    async def test_single_row(self):
        handler = _make_excerpt_handler(_make_log(10))
        result = await handler({"experience_id": "exp_001", "start_row": 5, "end_row": 5})
        data = _parse_result(result)
        assert len(data["rows"]) == 1
        assert data["rows"][0]["row"] == 5


class TestExcerptValidation:
    @pytest.mark.asyncio
    async def test_missing_experience_id(self):
        handler = _make_excerpt_handler()
        result = await handler({})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_not_found(self):
        handler = _make_excerpt_handler(None)
        result = await handler({"experience_id": "nonexistent"})
        assert _is_error(result)
        assert "not found" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_start_beyond_total(self):
        handler = _make_excerpt_handler(_make_log(3))
        result = await handler({"experience_id": "exp_001", "start_row": 10})
        assert _is_error(result)
        assert "exceeds" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# create_tool structure tests
# ---------------------------------------------------------------------------


class TestCreateToolStructure:
    def test_returns_three_tools(self):
        fs = MagicMock()
        tools = create_tool(fs_backend=fs)
        assert isinstance(tools, list)
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"recall_fetch", "recall_outline", "recall_excerpt"}
