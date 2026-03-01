"""Unit tests for the retriever tool — shared contract + variant-specific tests.

Both `baseline` and `rerank` variants are tested against the same MCP interface
contract using mocked KnowledgeStore and (for rerank) RerankerClient.
"""

from __future__ import annotations

import json
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from agenix.storage.models import CardType
from services.models import RerankResult
from tools.retriever.baseline.logic import create_tool as create_baseline_tool
from tools.retriever.rerank.logic import (
    _CANDIDATE_MULTIPLIER,
)
from tools.retriever.rerank.logic import (
    create_tool as create_rerank_tool,
)

RetrieverFixture = namedtuple("RetrieverFixture", ["handler", "store", "variant"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(
    card_id: str,
    title: str,
    content: str,
    card_type: str = "knowledge",
    distance: float = 0.5,
) -> dict:
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


def _parse_result(result: dict) -> dict:
    """Parse MCP tool result dict → JSON payload."""
    text = result["content"][0]["text"]
    return json.loads(text)


def _is_error(result: dict) -> bool:
    return result.get("is_error", False)


def _make_baseline(search_results: list[dict]) -> RetrieverFixture:
    """Create a baseline retriever with mocked store.

    The mock's search() respects the ``limit`` kwarg (like a real store)
    so that top_k-based truncation tests work correctly.
    """
    store = MagicMock()

    def _search(*, query, limit, card_type=None):
        return search_results[:limit]

    store.search.side_effect = _search
    sdk_tool = create_baseline_tool(knowledge_store=store)
    return RetrieverFixture(handler=sdk_tool.handler, store=store, variant="baseline")


def _make_rerank(
    search_results: list[dict],
    rerank_scores: list[float] | None = None,
) -> RetrieverFixture:
    """Create a rerank retriever with mocked store and reranker.

    If rerank_scores is None, scores mirror the baseline formula
    (1.0 - distance) so shared ordering tests pass for both variants.
    """
    store = MagicMock()
    store.search.return_value = search_results

    if rerank_scores is None:
        rerank_scores = [
            round(1.0 - r.get("_distance", 0.0), 4) for r in search_results
        ]

    reranker = AsyncMock()
    reranker.rank.return_value = RerankResult(
        scores=rerank_scores, model="test-model"
    )

    sdk_tool = create_rerank_tool(
        knowledge_store=store, reranker_client=reranker
    )
    return RetrieverFixture(handler=sdk_tool.handler, store=store, variant="rerank")


# ---------------------------------------------------------------------------
# Shared contract fixture — parametrized over both variants
# ---------------------------------------------------------------------------

# Default search results for shared tests: 25 cards across card types and
# distances. Ordered by increasing distance (most relevant first).
_DEFAULT_RESULTS = [
    # --- knowledge cards (GPU kernels) ---
    _make_search_result(
        "k01", "Triton tiling for GEMM",
        "Use BLOCK_M x BLOCK_N tiles with tl.dot", distance=0.02,
    ),
    _make_search_result(
        "k02", "CUDA shared memory padding",
        "Pad arrays to avoid bank conflicts", distance=0.05,
    ),
    _make_search_result(
        "k03", "Warp-level matrix ops",
        "Use wmma or mma.sync for tensor cores", distance=0.08,
    ),
    _make_search_result(
        "k04", "Register pressure in Triton",
        "Limit live variables per thread", distance=0.11,
    ),
    _make_search_result(
        "k05", "Coalesced global memory access",
        "Align thread accesses to cache lines", distance=0.14,
    ),
    _make_search_result(
        "k06", "Kernel fusion for elementwise ops",
        "Fuse ReLU with preceding matmul", distance=0.18,
    ),
    _make_search_result(
        "k07", "Async copy with cp.async",
        "Overlap compute with memory transfers", distance=0.22,
    ),
    _make_search_result(
        "k08", "Block size autotuning",
        "Search powers-of-2 block dimensions", distance=0.26,
    ),
    _make_search_result(
        "k09", "Triton program_id grid launch",
        "Ceil-divide N by BLOCK_SIZE for grid", distance=0.30,
    ),
    _make_search_result(
        "k10", "L2 cache persistence",
        "Use cudaStreamAttrValue for cache pinning", distance=0.34,
    ),
    # --- knowledge cards (algorithms) ---
    _make_search_result(
        "k11", "Dynamic programming patterns",
        "Use lru_cache for top-down DP", distance=0.38,
    ),
    _make_search_result(
        "k12", "Graph traversal BFS/DFS",
        "BFS for shortest path, DFS for cycle detection", distance=0.42,
    ),
    _make_search_result(
        "k13", "Binary search variants",
        "Use bisect_left for lower-bound queries", distance=0.46,
    ),
    _make_search_result(
        "k14", "Sliding window technique",
        "Maintain invariant while expanding window", distance=0.50,
    ),
    _make_search_result(
        "k15", "Union-Find with path compression",
        "Amortized near-O(1) connectivity queries", distance=0.54,
    ),
    # --- reflection cards ---
    _make_search_result(
        "r01", "Grid launch off-by-one bug",
        "Use ceil division for grid size",
        card_type="reflection", distance=0.58,
    ),
    _make_search_result(
        "r02", "dtype mismatch in accumulator",
        "Pin acc to float32 in mixed-precision",
        card_type="reflection", distance=0.62,
    ),
    _make_search_result(
        "r03", "Missing sync after shared mem write",
        "Add tl.debug_barrier or __syncthreads",
        card_type="reflection", distance=0.66,
    ),
    _make_search_result(
        "r04", "Incorrect reduction axis",
        "Verify axis param matches data layout",
        card_type="reflection", distance=0.70,
    ),
    _make_search_result(
        "r05", "Silent wraparound in int32 indexing",
        "Use tl.arange with int64 for large tensors",
        card_type="reflection", distance=0.74,
    ),
    # --- insight cards ---
    _make_search_result(
        "i01", "Memory-bound dominates workloads",
        "80% of kernels are memory-bound",
        card_type="insight", distance=0.78,
    ),
    _make_search_result(
        "i02", "Fusion yields largest speedups",
        "Fused kernels show >2x improvement",
        card_type="insight", distance=0.82,
    ),
    _make_search_result(
        "i03", "Autotuning beats manual tuning",
        "Grid search finds better params in 90% of cases",
        card_type="insight", distance=0.86,
    ),
    _make_search_result(
        "i04", "Correctness bugs cluster in indexing",
        "60% of bugs are off-by-one in grid/block",
        card_type="insight", distance=0.90,
    ),
    _make_search_result(
        "i05", "Larger models need fp32 accumulators",
        "Models >1B params diverge without fp32 acc",
        card_type="insight", distance=0.94,
    ),
]


@pytest.fixture(params=["baseline", "rerank"])
def retriever(request) -> RetrieverFixture:
    """Parametrized fixture producing a retriever for each variant."""
    if request.param == "baseline":
        return _make_baseline(list(_DEFAULT_RESULTS))
    else:
        return _make_rerank(list(_DEFAULT_RESULTS))


# ---------------------------------------------------------------------------
# Shared contract tests
# ---------------------------------------------------------------------------


class TestSharedContract:
    """Tests that both baseline and rerank must satisfy."""

    @pytest.mark.asyncio
    async def test_basic_query(self, retriever):
        """A valid query returns cards up to top_k."""
        result = await retriever.handler({"query": "test", "top_k": 10})
        parsed = _parse_result(result)
        assert len(parsed["cards"]) == 10

    @pytest.mark.asyncio
    async def test_output_fields(self, retriever):
        """Each card has exactly the required fields."""
        result = await retriever.handler({"query": "test", "top_k": 1})
        parsed = _parse_result(result)
        card = parsed["cards"][0]
        assert set(card.keys()) == {
            "card_id", "title", "content", "card_type", "score",
        }

    @pytest.mark.asyncio
    async def test_scores_descending(self, retriever):
        """Cards are ordered by score descending."""
        result = await retriever.handler({"query": "test", "top_k": 15})
        parsed = _parse_result(result)
        scores = [c["score"] for c in parsed["cards"]]
        assert len(scores) == 15
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, retriever):
        """top_k truncates output to fewer than available cards."""
        result = await retriever.handler({"query": "test", "top_k": 7})
        parsed = _parse_result(result)
        assert len(parsed["cards"]) == 7

    @pytest.mark.asyncio
    async def test_fewer_results_than_top_k(self):
        """When store returns fewer results than top_k, return all."""
        results = [
            _make_search_result("c1", "Card 1", "content 1", distance=0.2),
        ]
        for factory in [_make_baseline, _make_rerank]:
            r = factory(results)
            result = await r.handler({"query": "test", "top_k": 5})
            parsed = _parse_result(result)
            assert len(parsed["cards"]) == 1

    @pytest.mark.asyncio
    async def test_empty_query_error(self, retriever):
        """Empty query string returns error."""
        result = await retriever.handler({"query": ""})
        assert _is_error(result)
        assert "required" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_missing_query_error(self, retriever):
        """Missing query key returns error."""
        result = await retriever.handler({})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_card_type_filter(self):
        """card_type filter is forwarded to store.search()."""
        results = [
            _make_search_result(
                "c1", "Card 1", "content", card_type="reflection", distance=0.1
            ),
        ]
        for factory in [_make_baseline, _make_rerank]:
            r = factory(results)
            await r.handler({"query": "test", "card_type": "reflection"})
            call_kwargs = r.store.search.call_args.kwargs
            assert call_kwargs["card_type"] == CardType.REFLECTION

    @pytest.mark.asyncio
    async def test_invalid_card_type_error(self, retriever):
        """Invalid card_type returns error, store not called."""
        result = await retriever.handler(
            {"query": "test", "card_type": "invalid"}
        )
        assert _is_error(result)
        assert "Invalid card_type" in result["content"][0]["text"]
        retriever.store.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_store(self):
        """No results → empty cards list."""
        for factory in [_make_baseline, _make_rerank]:
            r = factory([])
            result = await r.handler({"query": "test"})
            parsed = _parse_result(result)
            assert parsed["cards"] == []

    @pytest.mark.asyncio
    async def test_score_rounding(self):
        """Scores are rounded to 4 decimal places."""
        # distance 0.123456789 → baseline score 0.8765
        results = [
            _make_search_result("c1", "Card 1", "content", distance=0.123456789),
        ]
        # For rerank, explicitly provide a score that needs rounding
        r_baseline = _make_baseline(results)
        result = await r_baseline.handler({"query": "test", "top_k": 1})
        parsed = _parse_result(result)
        score = parsed["cards"][0]["score"]
        assert score == round(score, 4)

        r_rerank = _make_rerank(results, rerank_scores=[0.876543211])
        result = await r_rerank.handler({"query": "test", "top_k": 1})
        parsed = _parse_result(result)
        assert parsed["cards"][0]["score"] == 0.8765

    @pytest.mark.asyncio
    async def test_default_top_k(self):
        """Default top_k is 5 — both variants accept omitted top_k."""
        results = [
            _make_search_result(f"c{i}", f"Card {i}", f"content {i}", distance=0.1 * i)
            for i in range(10)
        ]
        for factory in [_make_baseline, _make_rerank]:
            r = factory(results)
            result = await r.handler({"query": "test"})
            parsed = _parse_result(result)
            assert len(parsed["cards"]) == 5


# ---------------------------------------------------------------------------
# Baseline-specific tests
# ---------------------------------------------------------------------------


class TestBaselineSpecific:
    """Tests specific to the baseline (dense vector) variant."""

    @pytest.mark.asyncio
    async def test_score_is_similarity(self):
        """Baseline score = 1.0 - distance."""
        results = [
            _make_search_result("c1", "Card 1", "content", distance=0.25),
        ]
        r = _make_baseline(results)
        result = await r.handler({"query": "test", "top_k": 1})
        parsed = _parse_result(result)
        assert parsed["cards"][0]["score"] == 0.75

    @pytest.mark.asyncio
    async def test_search_limit_equals_top_k(self):
        """Baseline calls store.search with limit=top_k (no multiplier)."""
        r = _make_baseline([_make_search_result("c1", "Card 1", "content")])
        await r.handler({"query": "test", "top_k": 3})
        r.store.search.assert_called_once_with(
            query="test", limit=3, card_type=None
        )


# ---------------------------------------------------------------------------
# Rerank-specific tests
# ---------------------------------------------------------------------------


class TestRerankSpecific:
    """Tests specific to the rerank (two-stage) variant."""

    @pytest.mark.asyncio
    async def test_candidate_multiplier(self):
        """Rerank calls store.search with limit=5*top_k."""
        results = [_make_search_result("c1", "Card 1", "content")]
        r = _make_rerank(results, rerank_scores=[0.8])
        await r.handler({"query": "test", "top_k": 3})
        r.store.search.assert_called_once_with(
            query="test", limit=3 * _CANDIDATE_MULTIPLIER, card_type=None
        )

    @pytest.mark.asyncio
    async def test_reranker_not_called_on_empty(self):
        """Reranker.rank() is not called when store returns nothing."""
        store = MagicMock()
        store.search.return_value = []
        reranker = AsyncMock()
        sdk_tool = create_rerank_tool(
            knowledge_store=store, reranker_client=reranker
        )
        await sdk_tool.handler({"query": "test"})
        reranker.rank.assert_not_called()

    @pytest.mark.asyncio
    async def test_documents_are_card_contents(self):
        """Documents passed to reranker are card.content values."""
        results = [
            _make_search_result("c1", "Card 1", "Use tiling"),
            _make_search_result("c2", "Card 2", "Apply unrolling"),
        ]
        store = MagicMock()
        store.search.return_value = results
        reranker = AsyncMock()
        reranker.rank.return_value = RerankResult(
            scores=[0.9, 0.1], model="m"
        )
        sdk_tool = create_rerank_tool(
            knowledge_store=store, reranker_client=reranker
        )
        await sdk_tool.handler({"query": "matrix multiply", "top_k": 2})
        reranker.rank.assert_called_once_with(
            query="matrix multiply",
            documents=["Use tiling", "Apply unrolling"],
        )

    @pytest.mark.asyncio
    async def test_reranker_score_overrides_distance(self):
        """Reranker score determines order even when it disagrees with distance."""
        results = [
            _make_search_result("c1", "Card 1", "content 1", distance=0.1),
            _make_search_result("c2", "Card 2", "content 2", distance=0.9),
        ]
        # Reranker says c2 is more relevant despite high distance
        r = _make_rerank(results, rerank_scores=[0.3, 0.85])
        result = await r.handler({"query": "test", "top_k": 2})
        parsed = _parse_result(result)
        assert parsed["cards"][0]["card_id"] == "c2"
        assert parsed["cards"][0]["score"] == 0.85
        assert parsed["cards"][1]["card_id"] == "c1"
        assert parsed["cards"][1]["score"] == 0.3
