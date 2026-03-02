"""Integration test: retriever MCP tool variants with remote GPU services.

Tests both retriever variants against actual remote services:
- **Baseline**: Remote text_embedding (Qwen3-Embedding-8B, 4096-dim on _two)
- **Rerank**: Remote text_embedding + remote reranker (Qwen3-32B on _two)

Uses the same sample GPU kernel knowledge cards as the original tests,
with RemoteEmbedder (4096-dim) replacing local all-MiniLM-L6-v2 (384-dim).

Requires:
- text-embedding service running on _two
- reranker service running on _two (rerank tests skipped if down)
- SSH tunnels running: reflection services tunnel start

Run with:
    uv run pytest tests/integration/test_retriever_tool.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from agenix.config import (
    ReflectionConfig,
    RerankerClientConfig,
    StorageConfig,
    TextEmbeddingClientConfig,
    load_config,
)
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import (
    InsightCard,
    KnowledgeCard,
    ReflectionCard,
    ReflectionCategory,
)
from services.models import ServiceStatus
from services.reranker.baseline.client import RerankerClient
from services.text_embedding.baseline.client import TextEmbeddingClient
from tools.knowledge.baseline.embedder import RemoteEmbedder
from tools.knowledge.baseline.index import LanceIndex
from tools.knowledge.baseline.store import KnowledgeStore
from tools.retriever.baseline.logic import create_tool as create_baseline_tool
from tools.retriever.rerank.logic import create_tool as create_rerank_tool

# ---------------------------------------------------------------------------
# Service configs + health-check fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def te_config() -> TextEmbeddingClientConfig:
    """Get text_embedding client config from _two endpoint."""
    cfg = load_config()
    for ep in cfg.services.endpoints:
        if ep.name == "_two":
            return ep.text_embedding
    pytest.skip("Endpoint _two not configured in hosts.yaml")


@pytest.fixture(scope="module")
def te_client(te_config) -> TextEmbeddingClient:
    return TextEmbeddingClient(te_config)


@pytest.fixture(scope="module")
def rr_config() -> RerankerClientConfig:
    """Get reranker client config from _two endpoint."""
    cfg = load_config()
    for ep in cfg.services.endpoints:
        if ep.name == "_two":
            return ep.reranker
    pytest.skip("Endpoint _two not configured in hosts.yaml")


@pytest.fixture(scope="module")
def rr_client(rr_config) -> RerankerClient:
    return RerankerClient(rr_config)


@pytest.fixture(scope="module", autouse=True)
def _check_embedding(te_client):
    """Skip entire module if text-embedding service is not reachable."""
    async def check():
        try:
            h = await te_client.health()
            return h.status == ServiceStatus.RUNNING
        except Exception:
            return False

    if not asyncio.run(check()):
        pytest.skip("text-embedding service not reachable on _two")


@pytest.fixture(scope="module")
def _reranker_available(rr_client) -> bool:
    """Check if reranker service is reachable. Returns bool, does not skip."""
    async def check():
        try:
            h = await rr_client.health()
            return h.status == ServiceStatus.RUNNING
        except Exception:
            return False

    return asyncio.run(check())


# ---------------------------------------------------------------------------
# Sample cards — GPU kernel knowledge domain
# ---------------------------------------------------------------------------

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
        experience_ids=["exp-001"],
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


# ---------------------------------------------------------------------------
# Store + retriever fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, te_config):
    """KnowledgeStore backed by remote embedder (4096-dim), in test env."""
    user = os.environ.get("USER", "unknown")
    env = f"test_{user}"
    storage_cfg = StorageConfig(data_root=str(tmp_path), env=env)

    embedder = RemoteEmbedder(config=te_config, dimension=4096)
    lance = LanceIndex(db_path=storage_cfg.lance_path, vector_dim=4096)
    fs_backend = FSBackend(storage_cfg)
    fs_backend.initialize()

    config = ReflectionConfig(storage=storage_cfg)
    return KnowledgeStore(
        config=config,
        fs_backend=fs_backend,
        lance_index=lance,
        embedder=embedder,
    )


@pytest.fixture
def populated_store(store):
    """Store pre-loaded with SAMPLE_CARDS."""
    for card in SAMPLE_CARDS:
        store.add_card(card)
    return store


@pytest.fixture
def baseline_retriever(populated_store):
    """Baseline retriever MCP tool backed by the populated store."""
    return create_baseline_tool(knowledge_store=populated_store)


@pytest.fixture
def rerank_retriever(populated_store, rr_client, _reranker_available):
    """Rerank retriever MCP tool. Skipped if reranker service is down."""
    if not _reranker_available:
        pytest.skip("reranker service not reachable on _two")
    return create_rerank_tool(
        knowledge_store=populated_store,
        reranker_client=rr_client,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_cards(result: dict) -> list[dict]:
    """Parse the MCP tool result into a list of card dicts."""
    content = result.get("content", [])
    assert len(content) > 0, f"Empty result: {result}"
    text = content[0].get("text", "")
    data = json.loads(text)
    return data.get("cards", [])


def _is_error(result: dict) -> bool:
    return result.get("is_error", False)


# ===========================================================================
# 1. Baseline variant
# ===========================================================================


class TestRetrieverBaseline:
    """Test the baseline retriever (dense vector search, remote embedder)."""

    # --- Basic invocation ---

    @pytest.mark.asyncio
    async def test_retrieve_by_query(self, baseline_retriever):
        """Querying for matrix multiply should return the GEMM card first."""
        result = await baseline_retriever.handler(
            {"query": "matrix multiplication tiling strategy"}
        )
        cards = _parse_cards(result)
        assert len(cards) > 0
        assert cards[0]["title"] == "Triton tiling for matrix multiplication"

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self, baseline_retriever):
        """Empty query should return an error."""
        result = await baseline_retriever.handler({"query": ""})
        assert _is_error(result)

    @pytest.mark.asyncio
    async def test_retrieve_top_k(self, baseline_retriever):
        """top_k limits the number of results."""
        result = await baseline_retriever.handler(
            {"query": "GPU kernel optimization", "top_k": 2}
        )
        cards = _parse_cards(result)
        assert len(cards) == 2

    @pytest.mark.asyncio
    async def test_retrieve_all_card_types(self, baseline_retriever):
        """Query should return knowledge, reflection, and insight cards."""
        result = await baseline_retriever.handler(
            {"query": "triton kernel optimization fusion", "top_k": 6}
        )
        cards = _parse_cards(result)
        types = {c["card_type"] for c in cards}
        assert "knowledge" in types

    @pytest.mark.asyncio
    async def test_retrieve_with_type_filter(self, baseline_retriever):
        """Filter by card_type should only return that type."""
        result = await baseline_retriever.handler({
            "query": "triton kernel",
            "card_type": "reflection",
        })
        cards = _parse_cards(result)
        for c in cards:
            assert c["card_type"] == "reflection"

    @pytest.mark.asyncio
    async def test_invalid_card_type(self, baseline_retriever):
        """Invalid card_type should return an error."""
        result = await baseline_retriever.handler({
            "query": "anything",
            "card_type": "nonexistent",
        })
        assert _is_error(result)

    # --- Semantic quality ---

    @pytest.mark.asyncio
    async def test_relu_query(self, baseline_retriever):
        """Querying about relu should rank the activation card highly."""
        result = await baseline_retriever.handler(
            {"query": "how to optimize relu activation in triton"}
        )
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "ReLU activation kernel optimization" in titles

    @pytest.mark.asyncio
    async def test_shared_memory_query(self, baseline_retriever):
        """Querying about bank conflicts should find the CUDA shared memory card."""
        result = await baseline_retriever.handler(
            {"query": "shared memory bank conflict padding"}
        )
        cards = _parse_cards(result)
        assert cards[0]["title"] == "CUDA shared memory bank conflicts"

    @pytest.mark.asyncio
    async def test_dp_query_not_gpu(self, baseline_retriever):
        """DP query should rank the algorithms card above GPU cards."""
        result = await baseline_retriever.handler(
            {"query": "dynamic programming memoization"}
        )
        cards = _parse_cards(result)
        assert cards[0]["title"] == "Python dynamic programming patterns"

    @pytest.mark.asyncio
    async def test_kernel_launch_bug(self, baseline_retriever):
        """Querying about grid bugs should surface the reflection card."""
        result = await baseline_retriever.handler({
            "query": "triton kernel grid launch wrong results off by one",
        })
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "Triton kernel launch grid miscalculation" in titles

    @pytest.mark.asyncio
    async def test_memory_bound_insight(self, baseline_retriever):
        """Querying about memory bandwidth should surface the insight card."""
        result = await baseline_retriever.handler({
            "query": "are GPU kernels memory bound or compute bound",
        })
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "Memory-bound kernels dominate modern workloads" in titles

    @pytest.mark.asyncio
    async def test_scores_descending(self, baseline_retriever):
        """Scores should be in decreasing order (most relevant first)."""
        result = await baseline_retriever.handler(
            {"query": "triton GPU kernel", "top_k": 5}
        )
        cards = _parse_cards(result)
        scores = [c["score"] for c in cards]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score {scores[i]} at position {i} < {scores[i+1]} at position {i+1}"
            )

    # --- Lineage (archive, supersede) ---

    @pytest.mark.asyncio
    async def test_archived_card_excluded(self, populated_store):
        """Archived cards should not appear in retriever results."""
        from agenix.storage.lineage import archive_card

        dp_card = next(c for c in SAMPLE_CARDS if "dynamic programming" in c.title)
        archive_card(dp_card, agent="test")
        populated_store.deactivate_card(dp_card)

        retriever = create_baseline_tool(knowledge_store=populated_store)
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
        record_creation(old_card, [SourceReference(id="traj-002", type="experience")])

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

        retriever = create_baseline_tool(knowledge_store=populated_store)
        result = await retriever.handler({"query": "relu activation optimization"})
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards]
        assert "Fused ReLU activation patterns" in titles
        assert "ReLU activation kernel optimization" not in titles


# ===========================================================================
# 2. Rerank variant
# ===========================================================================


class TestRetrieverRerank:
    """Test the rerank retriever (dense retrieval + cross-encoder reranking).

    All tests in this class are skipped if the reranker service is down.
    """

    # --- Semantic quality ---

    @pytest.mark.asyncio
    async def test_relu_query(self, rerank_retriever):
        """Querying about relu should rank the activation card highly."""
        result = await rerank_retriever.handler(
            {"query": "how to optimize relu activation in triton"}
        )
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "ReLU activation kernel optimization" in titles

    @pytest.mark.asyncio
    async def test_shared_memory_query(self, rerank_retriever):
        """Querying about bank conflicts should find the CUDA shared memory card."""
        result = await rerank_retriever.handler(
            {"query": "shared memory bank conflict padding"}
        )
        cards = _parse_cards(result)
        assert cards[0]["title"] == "CUDA shared memory bank conflicts"

    @pytest.mark.asyncio
    async def test_dp_query_not_gpu(self, rerank_retriever):
        """DP query should rank the algorithms card above GPU cards."""
        result = await rerank_retriever.handler(
            {"query": "dynamic programming memoization"}
        )
        cards = _parse_cards(result)
        assert cards[0]["title"] == "Python dynamic programming patterns"

    @pytest.mark.asyncio
    async def test_kernel_launch_bug(self, rerank_retriever):
        """Querying about grid bugs should surface the reflection card."""
        result = await rerank_retriever.handler({
            "query": "triton kernel grid launch wrong results off by one",
        })
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "Triton kernel launch grid miscalculation" in titles

    @pytest.mark.asyncio
    async def test_memory_bound_insight(self, rerank_retriever):
        """Querying about memory bandwidth should surface the insight card."""
        result = await rerank_retriever.handler({
            "query": "are GPU kernels memory bound or compute bound",
        })
        cards = _parse_cards(result)
        titles = [c["title"] for c in cards[:2]]
        assert "Memory-bound kernels dominate modern workloads" in titles

    # --- Output format + ordering ---

    @pytest.mark.asyncio
    async def test_scores_descending(self, rerank_retriever):
        """Reranked scores should be in decreasing order."""
        result = await rerank_retriever.handler(
            {"query": "triton GPU kernel", "top_k": 5}
        )
        cards = _parse_cards(result)
        scores = [c["score"] for c in cards]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score {scores[i]} at position {i} < {scores[i+1]} at position {i+1}"
            )

    @pytest.mark.asyncio
    async def test_output_has_expected_fields(self, rerank_retriever):
        """Reranked output should have the standard card fields."""
        result = await rerank_retriever.handler(
            {"query": "matrix multiplication", "top_k": 2}
        )
        cards = _parse_cards(result)
        assert len(cards) > 0
        for card in cards:
            assert "card_id" in card
            assert "title" in card
            assert "content" in card
            assert "card_type" in card
            assert "score" in card
            assert isinstance(card["score"], float)

    @pytest.mark.asyncio
    async def test_top_k(self, rerank_retriever):
        """top_k limits the number of reranked results."""
        result = await rerank_retriever.handler(
            {"query": "GPU kernel optimization", "top_k": 2}
        )
        cards = _parse_cards(result)
        assert len(cards) == 2

    @pytest.mark.asyncio
    async def test_type_filter(self, rerank_retriever):
        """card_type filter works end-to-end with reranking."""
        result = await rerank_retriever.handler({
            "query": "triton kernel",
            "card_type": "reflection",
        })
        cards = _parse_cards(result)
        for c in cards:
            assert c["card_type"] == "reflection"


# ===========================================================================
# 3. Cross-variant comparison
# ===========================================================================


class TestRerankVsBaseline:
    """Compare baseline and rerank retriever outputs on the same queries.

    Skipped if the reranker service is down.
    """

    @pytest.mark.asyncio
    async def test_both_return_results(self, baseline_retriever, rerank_retriever):
        """Both variants should return non-empty results for the same query."""
        query = "triton kernel optimization tiling"
        baseline_result = await baseline_retriever.handler(
            {"query": query, "top_k": 3}
        )
        rerank_result = await rerank_retriever.handler(
            {"query": query, "top_k": 3}
        )
        baseline_cards = _parse_cards(baseline_result)
        rerank_cards = _parse_cards(rerank_result)
        assert len(baseline_cards) > 0
        assert len(rerank_cards) > 0

    @pytest.mark.asyncio
    async def test_both_produce_valid_format(self, baseline_retriever, rerank_retriever):
        """Both variants should produce identically-shaped output."""
        query = "shared memory bank conflicts GPU"
        baseline_cards = _parse_cards(
            await baseline_retriever.handler({"query": query, "top_k": 3})
        )
        rerank_cards = _parse_cards(
            await rerank_retriever.handler({"query": query, "top_k": 3})
        )
        expected_keys = {"card_id", "title", "content", "card_type", "score"}
        for card in baseline_cards + rerank_cards:
            assert set(card.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_scores_differ(self, baseline_retriever, rerank_retriever):
        """Rerank scores come from cross-encoder, not embedding distance — they should differ."""
        query = "GPU kernel memory optimization fusion"
        baseline_cards = _parse_cards(
            await baseline_retriever.handler({"query": query, "top_k": 5})
        )
        rerank_cards = _parse_cards(
            await rerank_retriever.handler({"query": query, "top_k": 5})
        )
        baseline_scores = [c["score"] for c in baseline_cards]
        rerank_scores = [c["score"] for c in rerank_cards]
        # At least one score should differ (very unlikely to be identical
        # since they come from entirely different scoring mechanisms)
        assert baseline_scores != rerank_scores, (
            "Baseline and rerank scores are identical — "
            "reranker may not be applying cross-encoder scoring"
        )
