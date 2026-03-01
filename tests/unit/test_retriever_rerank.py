"""Tests for the retriever tool (rerank variant) — two-stage retrieve + rerank."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.models import RerankResult
from tools.retriever.rerank.logic import _CANDIDATE_MULTIPLIER, create_tool


def _make_search_result(
    card_id: str,
    title: str,
    content: str,
    card_type: str = "knowledge",
    distance: float = 0.5,
):
    """Create a mock KnowledgeStore.search() result dict."""
    card = MagicMock()
    card.content = content
    return {
        "card_id": card_id,
        "title": title,
        "card_type": card_type,
        "domain": "general",
        "_distance": distance,
        "card": card,
    }


def _make_tool(search_results: list[dict], rerank_scores: list[float]):
    """Create the MCP tool handler with mocked store and reranker.

    Returns (handler, store_mock, reranker_mock) where handler is the
    async callable from SdkMcpTool.handler.
    """
    store = MagicMock()
    store.search.return_value = search_results

    reranker = AsyncMock()
    reranker.rank.return_value = RerankResult(scores=rerank_scores, model="test-model")

    sdk_tool = create_tool(knowledge_store=store, reranker_client=reranker)
    return sdk_tool.handler, store, reranker


def _parse_result(result: dict) -> dict:
    """Parse MCP tool result dict → JSON payload."""
    text = result["content"][0]["text"]
    return json.loads(text)


class TestRerankBasic:
    @pytest.mark.asyncio
    async def test_reranker_score_and_order(self):
        """Results are ordered by reranker score, not embedding distance."""
        results = [
            _make_search_result("c1", "Card 1", "content 1", distance=0.1),
            _make_search_result("c2", "Card 2", "content 2", distance=0.2),
            _make_search_result("c3", "Card 3", "content 3", distance=0.3),
        ]
        # Reranker reverses the order: c3 is most relevant, c1 least
        scores = [0.2, 0.5, 0.9]

        tool_fn, _, _ = _make_tool(results, scores)
        result = await tool_fn({"query": "test query", "top_k": 3})
        parsed = _parse_result(result)

        cards = parsed["cards"]
        assert len(cards) == 3
        assert cards[0]["card_id"] == "c3"
        assert cards[0]["score"] == 0.9
        assert cards[1]["card_id"] == "c2"
        assert cards[1]["score"] == 0.5
        assert cards[2]["card_id"] == "c1"
        assert cards[2]["score"] == 0.2

    @pytest.mark.asyncio
    async def test_top_k_truncation(self):
        """Only top_k results returned after reranking."""
        results = [
            _make_search_result(f"c{i}", f"Card {i}", f"content {i}")
            for i in range(5)
        ]
        scores = [0.1, 0.9, 0.3, 0.7, 0.5]

        tool_fn, _, _ = _make_tool(results, scores)
        result = await tool_fn({"query": "test", "top_k": 2})
        parsed = _parse_result(result)

        cards = parsed["cards"]
        assert len(cards) == 2
        assert cards[0]["card_id"] == "c1"  # score 0.9
        assert cards[1]["card_id"] == "c3"  # score 0.7


class TestCandidateMultiplier:
    @pytest.mark.asyncio
    async def test_search_called_with_5k(self):
        """Store.search is called with limit=5*top_k."""
        results = [_make_search_result("c1", "Card 1", "content 1")]
        scores = [0.8]

        tool_fn, store, _ = _make_tool(results, scores)
        await tool_fn({"query": "test", "top_k": 3})

        store.search.assert_called_once_with(
            query="test", limit=3 * _CANDIDATE_MULTIPLIER, card_type=None
        )

    @pytest.mark.asyncio
    async def test_default_top_k(self):
        """Default top_k=5, so search is called with limit=25."""
        results = [_make_search_result("c1", "Card 1", "content 1")]
        scores = [0.8]

        tool_fn, store, _ = _make_tool(results, scores)
        await tool_fn({"query": "test"})

        store.search.assert_called_once_with(
            query="test", limit=5 * _CANDIDATE_MULTIPLIER, card_type=None
        )


class TestFewerResults:
    @pytest.mark.asyncio
    async def test_fewer_results_than_5k(self):
        """When store returns fewer results than 5*K, return all after reranking."""
        results = [
            _make_search_result("c1", "Card 1", "content 1"),
            _make_search_result("c2", "Card 2", "content 2"),
        ]
        scores = [0.6, 0.9]

        tool_fn, _, _ = _make_tool(results, scores)
        result = await tool_fn({"query": "test", "top_k": 5})
        parsed = _parse_result(result)

        cards = parsed["cards"]
        assert len(cards) == 2
        assert cards[0]["card_id"] == "c2"
        assert cards[1]["card_id"] == "c1"


class TestEmptyQuery:
    @pytest.mark.asyncio
    async def test_empty_string(self):
        """Empty query returns error."""
        tool_fn, _, _ = _make_tool([], [])
        result = await tool_fn({"query": ""})
        assert result["is_error"] is True
        assert "required" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_query(self):
        """Missing query key returns error."""
        tool_fn, _, _ = _make_tool([], [])
        result = await tool_fn({})
        assert result["is_error"] is True


class TestCardTypeFilter:
    @pytest.mark.asyncio
    async def test_card_type_passed_to_store(self):
        """card_type filter is forwarded to store.search()."""
        results = [_make_search_result("c1", "Card 1", "content 1", card_type="reflection")]
        scores = [0.7]

        tool_fn, store, _ = _make_tool(results, scores)
        await tool_fn({"query": "test", "card_type": "reflection"})

        from agenix.storage.models import CardType
        store.search.assert_called_once_with(
            query="test", limit=5 * _CANDIDATE_MULTIPLIER, card_type=CardType.REFLECTION
        )

    @pytest.mark.asyncio
    async def test_invalid_card_type(self):
        """Invalid card_type returns error without calling store."""
        tool_fn, store, _ = _make_tool([], [])
        result = await tool_fn({"query": "test", "card_type": "invalid"})

        assert result["is_error"] is True
        assert "Invalid card_type" in result["content"][0]["text"]
        store.search.assert_not_called()


class TestEmptyStore:
    @pytest.mark.asyncio
    async def test_no_results(self):
        """Empty store returns empty cards list, reranker not called."""
        tool_fn, _, reranker = _make_tool([], [])
        result = await tool_fn({"query": "test"})
        parsed = _parse_result(result)

        assert parsed["cards"] == []
        reranker.rank.assert_not_called()


class TestScoreFromReranker:
    @pytest.mark.asyncio
    async def test_scores_are_reranker_scores(self):
        """Returned scores are reranker scores, not embedding distances."""
        results = [
            _make_search_result("c1", "Card 1", "content 1", distance=0.1),
            _make_search_result("c2", "Card 2", "content 2", distance=0.9),
        ]
        # Reranker gives c2 a high score despite high distance
        scores = [0.3, 0.85]

        tool_fn, _, _ = _make_tool(results, scores)
        result = await tool_fn({"query": "test", "top_k": 2})
        parsed = _parse_result(result)

        cards = parsed["cards"]
        # c2 has higher reranker score
        assert cards[0]["card_id"] == "c2"
        assert cards[0]["score"] == 0.85
        assert cards[1]["card_id"] == "c1"
        assert cards[1]["score"] == 0.3

    @pytest.mark.asyncio
    async def test_score_rounding(self):
        """Scores are rounded to 4 decimal places."""
        results = [_make_search_result("c1", "Card 1", "content 1")]
        scores = [0.123456789]

        tool_fn, _, _ = _make_tool(results, scores)
        result = await tool_fn({"query": "test", "top_k": 1})
        parsed = _parse_result(result)

        assert parsed["cards"][0]["score"] == 0.1235


class TestRerankerInput:
    @pytest.mark.asyncio
    async def test_documents_are_card_contents(self):
        """Documents passed to reranker are card.content values."""
        results = [
            _make_search_result("c1", "Card 1", "Use tiling for cache locality"),
            _make_search_result("c2", "Card 2", "Apply loop unrolling"),
        ]
        scores = [0.9, 0.1]

        tool_fn, _, reranker = _make_tool(results, scores)
        await tool_fn({"query": "matrix multiply", "top_k": 2})

        reranker.rank.assert_called_once_with(
            query="matrix multiply",
            documents=["Use tiling for cache locality", "Apply loop unrolling"],
        )

    @pytest.mark.asyncio
    async def test_output_fields(self):
        """Each card in output has all required fields."""
        results = [
            _make_search_result("c1", "Card 1", "content 1", card_type="knowledge"),
        ]
        scores = [0.75]

        tool_fn, _, _ = _make_tool(results, scores)
        result = await tool_fn({"query": "test", "top_k": 1})
        parsed = _parse_result(result)

        card = parsed["cards"][0]
        assert set(card.keys()) == {"card_id", "title", "content", "card_type", "score"}
        assert card["card_id"] == "c1"
        assert card["title"] == "Card 1"
        assert card["content"] == "content 1"
        assert card["card_type"] == "knowledge"
        assert card["score"] == 0.75
