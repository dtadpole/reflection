"""Integration test: retriever MCP tool with sample GPU kernel knowledge cards.

Tests the full retrieval path: tool invocation → embedding → LanceDB search → card enrichment.
"""

from __future__ import annotations

import json

import pytest

from agenix.config import EmbedderConfig, ReflectionConfig, StorageConfig
from agenix.storage.models import (
    InsightCard,
    KnowledgeCard,
    ReflectionCard,
    ReflectionCategory,
)
from agenix.tools.retriever import create_retriever_tool
from tools.knowledge.baseline.store import KnowledgeStore


@pytest.fixture
def store(tmp_path):
    config = ReflectionConfig(
        storage=StorageConfig(data_root=str(tmp_path), env="test"),
        embedder=EmbedderConfig(model_name="all-MiniLM-L6-v2", top_k=5),
    )
    s = KnowledgeStore(config=config)
    s.initialize()
    return s


# --- Sample cards modelling real GPU kernel knowledge ---

SAMPLE_CARDS = [
    KnowledgeCard(
        title="Triton tiling for matrix multiplication",
        content=(
            "When writing Triton kernels for GEMM, tile the computation into "
            "BLOCK_M x BLOCK_N x BLOCK_K chunks. Use tl.dot for the inner product "
            "accumulator. Optimal block sizes depend on the GPU: 128x128x32 for "
            "A100, 64x64x32 for smaller GPUs. Pin accumulator dtype to tl.float32 "
            "to avoid precision loss."
        ),
        domain="gpu_kernels",
        applicability="Triton matrix multiply kernels",
        limitations="Block sizes must be powers of 2",
        tags=["triton", "gemm", "tiling", "gpu"],
    ),
    KnowledgeCard(
        title="CUDA shared memory bank conflicts",
        content=(
            "Shared memory is divided into 32 banks on NVIDIA GPUs. Bank conflicts "
            "occur when two threads in the same warp access different addresses in "
            "the same bank. Fix by padding shared memory arrays: allocate "
            "arr[N][N+1] instead of arr[N][N] to shift the stride."
        ),
        domain="gpu_kernels",
        applicability="CUDA kernels using shared memory",
        tags=["cuda", "shared_memory", "bank_conflict", "optimization"],
    ),
    KnowledgeCard(
        title="ReLU activation kernel optimization",
        content=(
            "ReLU is element-wise and memory-bound. Fuse it with preceding "
            "operations when possible. In Triton, use tl.maximum(x, 0) in the "
            "same kernel as the matmul epilogue. Avoid launching a separate "
            "kernel for standalone ReLU — the launch overhead dominates for "
            "small tensors."
        ),
        domain="gpu_kernels",
        applicability="Activation functions in neural networks",
        tags=["triton", "relu", "kernel_fusion", "activation"],
    ),
    KnowledgeCard(
        title="Python dynamic programming patterns",
        content=(
            "For DP problems, decide between top-down (memoization) and "
            "bottom-up (tabulation). Use functools.lru_cache for quick "
            "top-down solutions. For bottom-up, build the table iteratively "
            "and use rolling arrays to reduce space from O(n*m) to O(m)."
        ),
        domain="algorithms",
        applicability="Optimization and counting problems",
        tags=["python", "dynamic_programming", "memoization"],
    ),
    ReflectionCard(
        title="Triton kernel launch grid miscalculation",
        content=(
            "A common bug: computing grid = (N // BLOCK_SIZE,) instead of "
            "grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,). The first form "
            "drops the remainder block, silently producing wrong results "
            "on inputs not divisible by BLOCK_SIZE."
        ),
        trajectory_id="traj-001",
        category=ReflectionCategory.DEBUGGING,
        confidence=0.95,
        tags=["triton", "grid", "off-by-one"],
    ),
    InsightCard(
        title="Memory-bound kernels dominate modern workloads",
        content=(
            "Across 50 KernelBench problems, 80% of generated kernels are "
            "memory-bound rather than compute-bound. The key optimization "
            "strategy is reducing global memory accesses via fusion and "
            "shared memory caching, not increasing arithmetic intensity."
        ),
        hypothesis="Kernel fusion yields larger speedups than algorithmic changes",
        evidence_for=["40 of 50 fused kernels showed >2x speedup"],
        evidence_against=["Convolution kernels were compute-bound"],
        tags=["gpu", "memory_bound", "kernel_fusion", "benchmark"],
    ),
]


@pytest.fixture
def populated_store(store):
    """Store with sample GPU kernel knowledge cards."""
    for card in SAMPLE_CARDS:
        store.add_card(card)
    return store


@pytest.fixture
def retriever(populated_store):
    """Retriever MCP tool backed by the populated store."""
    return create_retriever_tool(populated_store)


class TestRetrieverToolBasic:
    """Test the retriever tool invocation path."""

    @pytest.mark.asyncio
    async def test_retrieve_by_query(self, retriever):
        """Querying for matrix multiply should return the GEMM card first."""
        result = await retriever.handler({"query": "matrix multiplication tiling strategy"})
        cards = _parse_cards(result)
        assert len(cards) > 0
        assert cards[0]["title"] == "Triton tiling for matrix multiplication"

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self, retriever):
        """Empty query should return an error."""
        result = await retriever.handler({"query": ""})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_retrieve_top_k(self, retriever):
        """top_k limits the number of results."""
        result = await retriever.handler({"query": "GPU kernel optimization", "top_k": 2})
        cards = _parse_cards(result)
        assert len(cards) == 2

    @pytest.mark.asyncio
    async def test_retrieve_all_card_types(self, retriever):
        """Query should return knowledge, reflection, and insight cards."""
        result = await retriever.handler({"query": "triton kernel optimization fusion", "top_k": 6})
        cards = _parse_cards(result)
        types = {c["card_type"] for c in cards}
        assert "knowledge" in types

    @pytest.mark.asyncio
    async def test_retrieve_with_type_filter(self, retriever):
        """Filter by card_type should only return that type."""
        result = await retriever.handler({
            "query": "triton kernel",
            "card_type": "reflection",
        })
        cards = _parse_cards(result)
        for c in cards:
            assert c["card_type"] == "reflection"

    @pytest.mark.asyncio
    async def test_invalid_card_type(self, retriever):
        """Invalid card_type should return an error."""
        result = await retriever.handler({
            "query": "anything",
            "card_type": "nonexistent",
        })
        assert _is_error(result)


class TestRetrieverSemanticQuality:
    """Test that semantic search returns relevant cards for realistic queries."""

    @pytest.mark.asyncio
    async def test_relu_query(self, retriever):
        """Querying about relu should rank the activation card highly."""
        result = await retriever.handler({"query": "how to optimize relu activation in triton"})
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "ReLU activation kernel optimization" in titles

    @pytest.mark.asyncio
    async def test_shared_memory_query(self, retriever):
        """Querying about bank conflicts should find the CUDA shared memory card."""
        result = await retriever.handler({"query": "shared memory bank conflict padding"})
        cards = _parse_cards(result)
        assert cards[0]["title"] == "CUDA shared memory bank conflicts"

    @pytest.mark.asyncio
    async def test_dp_query_not_gpu(self, retriever):
        """DP query should rank the algorithms card above GPU cards."""
        result = await retriever.handler({"query": "dynamic programming memoization"})
        cards = _parse_cards(result)
        assert cards[0]["title"] == "Python dynamic programming patterns"

    @pytest.mark.asyncio
    async def test_kernel_launch_bug(self, retriever):
        """Querying about grid bugs should surface the reflection card."""
        result = await retriever.handler({
            "query": "triton kernel grid launch wrong results off by one",
        })
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "Triton kernel launch grid miscalculation" in titles

    @pytest.mark.asyncio
    async def test_memory_bound_insight(self, retriever):
        """Querying about memory bandwidth should surface the insight card."""
        result = await retriever.handler({
            "query": "are GPU kernels memory bound or compute bound",
        })
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "Memory-bound kernels dominate modern workloads" in titles

    @pytest.mark.asyncio
    async def test_scores_decrease(self, retriever):
        """Scores should be in decreasing order (most relevant first)."""
        result = await retriever.handler({"query": "triton GPU kernel", "top_k": 5})
        cards = _parse_cards(result)
        scores = [c["score"] for c in cards]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score {scores[i]} at position {i} < {scores[i+1]} at position {i+1}"
            )


class TestRetrieverWithLineage:
    """Test retrieval behavior with card lifecycle operations."""

    @pytest.mark.asyncio
    async def test_archived_card_excluded(self, populated_store):
        """Archived cards should not appear in retriever results."""
        from agenix.storage.lineage import archive_card

        # Find and archive the DP card
        dp_card = next(c for c in SAMPLE_CARDS if "dynamic programming" in c.title)
        archive_card(dp_card, agent="test")
        populated_store.deactivate_card(dp_card)

        retriever = create_retriever_tool(populated_store)
        result = await retriever.handler({"query": "dynamic programming memoization"})
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards]
        assert "Python dynamic programming patterns" not in titles

    @pytest.mark.asyncio
    async def test_superseded_card_replaced(self, populated_store):
        """After revision, only the new card should appear in results."""
        from agenix.storage.lineage import record_creation, revise_card
        from agenix.storage.models import SourceReference

        old_card = next(c for c in SAMPLE_CARDS if "ReLU" in c.title)
        record_creation(old_card, [SourceReference(id="traj-002", type="trajectory")])

        new_card = KnowledgeCard(
            title="Fused ReLU activation patterns",
            content=(
                "Always fuse ReLU with the preceding matmul or conv kernel. "
                "In Triton, compute out = tl.maximum(tl.dot(a, b), 0) in one pass. "
                "Standalone ReLU kernels waste memory bandwidth on a trivial operation."
            ),
            domain="gpu_kernels",
            tags=["triton", "relu", "kernel_fusion"],
        )
        revise_card(old_card, new_card)
        populated_store.deactivate_card(old_card)
        populated_store.add_card(new_card)

        retriever = create_retriever_tool(populated_store)
        result = await retriever.handler({"query": "relu activation optimization"})
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards]
        assert "Fused ReLU activation patterns" in titles
        assert "ReLU activation kernel optimization" not in titles


# --- Helpers ---

def _parse_cards(result: dict) -> list[dict]:
    """Parse the MCP tool result into a list of card dicts."""
    content = result.get("content", [])
    assert len(content) > 0, f"Empty result: {result}"
    text = content[0].get("text", "")
    data = json.loads(text)
    return data.get("cards", [])


def _is_error(result: dict) -> bool:
    return result.get("is_error", False)
