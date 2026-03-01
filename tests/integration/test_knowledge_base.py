"""Integration test: knowledge base CRUD with remote GPU embeddings.

Tests all card lifecycle operations (create, revise, merge, split, archive)
and verifies consistency between:
- Filesystem (JSON files, queried via DuckDB)
- Embedding DB (LanceDB vector index)

Uses the remote text_embedding service on _two (Qwen3-Embedding-8B, 4096-dim)
via RemoteEmbedder, targeting the test_${USER} environment.

Requires:
- text-embedding service running on _two
- SSH tunnels running: reflection services tunnel start

Run with:
    uv run pytest tests/integration/test_knowledge_base.py -v -s
"""

from __future__ import annotations

import os

import pytest

from agenix.config import (
    ReflectionConfig,
    StorageConfig,
    TextEmbeddingClientConfig,
    load_config,
)
from agenix.storage.fs_backend import FSBackend
from agenix.storage.lineage import (
    archive_card,
    merge_cards,
    record_creation,
    revise_card,
    split_card,
)
from agenix.storage.models import (
    CardStatus,
    InsightCard,
    KnowledgeCard,
    LineageOperation,
    ReflectionCard,
    ReflectionCategory,
    SourceReference,
)
from services.models import ServiceStatus
from services.text_embedding.baseline.client import TextEmbeddingClient
from tools.knowledge.baseline.embedder import RemoteEmbedder
from tools.knowledge.baseline.index import LanceIndex
from tools.knowledge.baseline.store import KnowledgeStore

# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture(scope="module", autouse=True)
def _check_service(te_client):
    """Skip entire module if text-embedding service is not reachable."""
    import asyncio

    async def check():
        try:
            h = await te_client.health()
            return h.status == ServiceStatus.RUNNING
        except Exception:
            return False

    if not asyncio.run(check()):
        pytest.skip("text-embedding service not reachable on _two")


@pytest.fixture
def store(tmp_path, te_config):
    """Create a KnowledgeStore backed by remote embedder, in test env."""
    user = os.environ.get("USER", "unknown")
    env = f"test_{user}"
    storage_cfg = StorageConfig(data_root=str(tmp_path), env=env)

    embedder = RemoteEmbedder(config=te_config, dimension=4096)
    lance = LanceIndex(db_path=storage_cfg.lance_path, vector_dim=4096)
    fs = FSBackend(storage_cfg)
    fs.initialize()

    config = ReflectionConfig(storage=storage_cfg)
    return KnowledgeStore(
        config=config,
        fs_backend=fs,
        lance_index=lance,
        embedder=embedder,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_in_lance(store: KnowledgeStore, card_id: str) -> None:
    """Assert card is present in LanceDB index."""
    # Search with a dummy vector — just check the card is indexed
    count_before = store._lance.count()
    assert count_before > 0, "LanceDB is empty"
    results = store._lance.table.search().where(
        f"card_id = '{card_id}'"
    ).limit(1).to_list()
    assert len(results) == 1, f"Card {card_id} not found in LanceDB"


def _assert_not_in_lance(store: KnowledgeStore, card_id: str) -> None:
    """Assert card is NOT present in LanceDB index."""
    results = store._lance.table.search().where(
        f"card_id = '{card_id}'"
    ).limit(1).to_list()
    assert len(results) == 0, f"Card {card_id} should not be in LanceDB"


def _assert_fs_status(store: KnowledgeStore, card_id: str, expected: CardStatus) -> None:
    """Assert card exists on filesystem with expected status."""
    card = store.get_card(card_id)
    assert card is not None, f"Card {card_id} not found on filesystem"
    assert card.status == expected, (
        f"Card {card_id} status: expected {expected.value}, got {card.status.value}"
    )


def _assert_fs_count(store: KnowledgeStore, expected: int, **kwargs) -> None:
    """Assert card count on filesystem matches expected."""
    cards = store.list_cards(include_superseded=True, **kwargs)
    assert len(cards) == expected, (
        f"Expected {expected} cards, got {len(cards)}"
    )


def _assert_lance_count(store: KnowledgeStore, expected: int) -> None:
    """Assert LanceDB row count matches expected."""
    actual = store._lance.count()
    assert actual == expected, (
        f"LanceDB count: expected {expected}, got {actual}"
    )


# ===========================================================================
# 1. Create operations
# ===========================================================================


class TestCreate:
    """Test card creation across all card types."""

    def test_create_knowledge_card(self, store):
        """Creating a KnowledgeCard should persist to FS and index in LanceDB."""
        card = KnowledgeCard(
            title="GPU Memory Coalescing",
            content="Coalesced memory access patterns improve GPU bandwidth. "
            "Adjacent threads should access adjacent memory locations.",
            tags=["gpu", "memory", "optimization"],
            domain="gpu_optimization",
            applicability="CUDA and Triton kernels with global memory access",
        )
        record_creation(
            card,
            [SourceReference(id="traj-001", type="trajectory")],
            agent="test",
            run_tag="test_run",
        )
        store.add_card(card)

        # Filesystem check
        _assert_fs_status(store, card.card_id, CardStatus.ACTIVE)
        loaded = store.get_card(card.card_id)
        assert loaded.title == "GPU Memory Coalescing"
        assert loaded.domain == "gpu_optimization"
        assert len(loaded.lineage) == 1
        assert loaded.lineage[0].operation == LineageOperation.CREATE

        # Source refs preserved after round-trip
        assert len(loaded.source_refs) == 1
        assert loaded.source_refs[0].id == "traj-001"
        assert loaded.source_refs[0].type == "trajectory"

        # LanceDB check
        _assert_in_lance(store, card.card_id)

        # DuckDB query check
        results = store.fs.query_cards(f"card_id = '{card.card_id}'")
        assert len(results) == 1
        assert results[0]["title"] == "GPU Memory Coalescing"
        assert results[0]["status"] == "active"

    def test_create_insight_card(self, store):
        """Creating an InsightCard should persist and be searchable."""
        card = InsightCard(
            title="Warp Divergence Hypothesis",
            content="Warp divergence in conditional branches reduces SM utilization.",
            tags=["gpu", "warp", "performance"],
            hypothesis="Eliminating branch divergence improves kernel throughput by 2x",
        )
        record_creation(card, [], agent="test")
        store.add_card(card)

        _assert_fs_status(store, card.card_id, CardStatus.ACTIVE)
        _assert_in_lance(store, card.card_id)

        loaded = store.get_card(card.card_id)
        assert isinstance(loaded, InsightCard)
        assert loaded.hypothesis_status.value == "proposed"

    def test_create_reflection_card(self, store):
        """Creating a ReflectionCard should persist and be searchable."""
        card = ReflectionCard(
            title="Shared Memory Tiling Pattern",
            content="Using shared memory tiles for matrix multiply reduces "
            "global memory access by factor of tile_size.",
            tags=["gpu", "shared_memory", "tiling"],
            trajectory_id="traj-002",
            category=ReflectionCategory.OPTIMIZATION,
            confidence=0.85,
            supporting_steps=[3, 5, 7],
        )
        record_creation(
            card,
            [SourceReference(id="traj-002", type="trajectory")],
            agent="critic",
        )
        store.add_card(card)

        _assert_fs_status(store, card.card_id, CardStatus.ACTIVE)
        _assert_in_lance(store, card.card_id)

        loaded = store.get_card(card.card_id)
        assert isinstance(loaded, ReflectionCard)
        assert loaded.category == ReflectionCategory.OPTIMIZATION
        assert loaded.confidence == 0.85

    def test_create_multiple_and_search(self, store):
        """Multiple cards should all be searchable with semantic relevance."""
        cards = [
            KnowledgeCard(
                title="Triton Block Pointers",
                content="Triton block pointers enable efficient tensor access patterns "
                "with automatic bounds checking.",
                tags=["triton", "pointers"],
                domain="triton",
            ),
            KnowledgeCard(
                title="CUDA Thread Indexing",
                content="threadIdx, blockIdx, blockDim compute global thread ID for "
                "parallel work distribution.",
                tags=["cuda", "threads"],
                domain="cuda",
            ),
            KnowledgeCard(
                title="Python List Comprehension",
                content="List comprehensions provide concise syntax for creating lists "
                "from iterables with optional filtering.",
                tags=["python", "syntax"],
                domain="python",
            ),
        ]
        for c in cards:
            record_creation(c, [], agent="test")
            store.add_card(c)

        # Semantic search — triton query should find triton card first
        results = store.search("triton block pointer tensor access", limit=3)
        assert len(results) >= 1
        assert results[0]["card_id"] == cards[0].card_id

        # Domain filter
        results = store.search("programming", domain="python", limit=5)
        for r in results:
            assert r["domain"] == "python"


# ===========================================================================
# 2. Revise operations
# ===========================================================================


class TestRevise:
    """Test card revision: old card superseded, new card active."""

    def test_revise_updates_both_stores(self, store):
        """Revising a card should supersede old in FS, remove from Lance, add new."""
        old = KnowledgeCard(
            title="Loop Unrolling",
            content="Manually unrolling loops can improve instruction-level parallelism.",
            tags=["optimization"],
            domain="compiler",
        )
        record_creation(
            old,
            [SourceReference(id="traj-010", type="trajectory")],
            agent="organizer",
        )
        store.add_card(old)
        _assert_in_lance(store, old.card_id)

        # Revise with richer content
        new = KnowledgeCard(
            title="Loop Unrolling",
            content="Loop unrolling reduces branch overhead and enables ILP. "
            "Compiler pragmas (#pragma unroll) or Triton tl.static_range "
            "can automate this. Trade-off: register pressure increases.",
            tags=["optimization", "triton", "compiler"],
            domain="compiler",
        )
        revise_card(
            old,
            new,
            new_source_refs=[SourceReference(id="traj-011", type="trajectory")],
            agent="organizer",
            run_tag="run_revise",
        )
        store.deactivate_card(old)
        store.add_card(new)

        # Old card: superseded on FS, removed from Lance
        _assert_fs_status(store, old.card_id, CardStatus.SUPERSEDED)
        _assert_not_in_lance(store, old.card_id)
        old_loaded = store.get_card(old.card_id)
        assert old_loaded.superseded_by == new.card_id

        # New card: active on FS, present in Lance
        _assert_fs_status(store, new.card_id, CardStatus.ACTIVE)
        _assert_in_lance(store, new.card_id)
        new_loaded = store.get_card(new.card_id)
        assert old.card_id in new_loaded.predecessor_ids
        assert any(e.operation == LineageOperation.REVISE for e in new_loaded.lineage)

        # Source refs inherited from old + new
        source_ids = {ref.id for ref in new_loaded.source_refs}
        assert "traj-010" in source_ids
        assert "traj-011" in source_ids

        # DuckDB sees both cards with correct statuses
        old_rows = store.fs.query_cards(f"card_id = '{old.card_id}'")
        assert len(old_rows) == 1
        assert old_rows[0]["status"] == "superseded"
        new_rows = store.fs.query_cards(f"card_id = '{new.card_id}'")
        assert len(new_rows) == 1
        assert new_rows[0]["status"] == "active"

        # Search should find new card, not old
        results = store.search("loop unrolling optimization", limit=5)
        result_ids = {r["card_id"] for r in results}
        assert new.card_id in result_ids
        assert old.card_id not in result_ids

    def test_revise_preserves_lineage_chain(self, store):
        """Two successive revisions should form a lineage chain."""
        v1 = KnowledgeCard(
            title="Softmax Kernel",
            content="Basic softmax implementation.",
            domain="kernels",
        )
        record_creation(v1, [], agent="test")
        store.add_card(v1)

        v2 = KnowledgeCard(
            title="Softmax Kernel",
            content="Softmax with online normalization for numerical stability.",
            domain="kernels",
        )
        revise_card(v1, v2, agent="organizer")
        store.deactivate_card(v1)
        store.add_card(v2)

        v3 = KnowledgeCard(
            title="Softmax Kernel",
            content="Flash-attention style softmax: online normalization + tiling "
            "for O(1) extra memory.",
            domain="kernels",
        )
        revise_card(v2, v3, agent="organizer")
        store.deactivate_card(v2)
        store.add_card(v3)

        # Only v3 should be in Lance
        _assert_not_in_lance(store, v1.card_id)
        _assert_not_in_lance(store, v2.card_id)
        _assert_in_lance(store, v3.card_id)

        # All three on FS
        _assert_fs_status(store, v1.card_id, CardStatus.SUPERSEDED)
        _assert_fs_status(store, v2.card_id, CardStatus.SUPERSEDED)
        _assert_fs_status(store, v3.card_id, CardStatus.ACTIVE)

        # Chain: v3 → v2 → v1
        v3_loaded = store.get_card(v3.card_id)
        assert v2.card_id in v3_loaded.predecessor_ids
        v2_loaded = store.get_card(v2.card_id)
        assert v1.card_id in v2_loaded.predecessor_ids

        # Only active cards in default list
        active = store.list_cards(domain="kernels")
        active_ids = {c.card_id for c in active}
        assert v3.card_id in active_ids
        assert v1.card_id not in active_ids
        assert v2.card_id not in active_ids


# ===========================================================================
# 3. Merge operations
# ===========================================================================


class TestMerge:
    """Test merging multiple cards into one."""

    def test_merge_two_cards(self, store):
        """Merging two cards: both superseded, merged card active."""
        card_a = KnowledgeCard(
            title="Reduction: Sum",
            content="Parallel reduction for sum: tree-based reduction within warps.",
            tags=["reduction", "sum"],
            domain="gpu_patterns",
        )
        card_b = KnowledgeCard(
            title="Reduction: Max",
            content="Parallel reduction for max: same tree pattern, different operator.",
            tags=["reduction", "max"],
            domain="gpu_patterns",
        )
        record_creation(
            card_a,
            [SourceReference(id="traj-020", type="trajectory")],
            agent="critic",
        )
        record_creation(
            card_b,
            [SourceReference(id="traj-021", type="trajectory")],
            agent="critic",
        )
        store.add_card(card_a)
        store.add_card(card_b)

        # Merge into unified reduction card
        merged = KnowledgeCard(
            title="Parallel Reduction Patterns",
            content="Tree-based parallel reduction works for any associative operator "
            "(sum, max, min, product). Each level halves active threads. "
            "Warp-level shuffle instructions avoid shared memory for final steps.",
            tags=["reduction", "sum", "max", "warp_shuffle"],
            domain="gpu_patterns",
        )
        merge_cards([card_a, card_b], merged, agent="organizer", run_tag="run_merge")
        store.deactivate_card(card_a)
        store.deactivate_card(card_b)
        store.add_card(merged)

        # Source cards: superseded on FS, removed from Lance
        _assert_fs_status(store, card_a.card_id, CardStatus.SUPERSEDED)
        _assert_fs_status(store, card_b.card_id, CardStatus.SUPERSEDED)
        _assert_not_in_lance(store, card_a.card_id)
        _assert_not_in_lance(store, card_b.card_id)

        # Merged card: active on FS, in Lance
        _assert_fs_status(store, merged.card_id, CardStatus.ACTIVE)
        _assert_in_lance(store, merged.card_id)

        # Lineage
        merged_loaded = store.get_card(merged.card_id)
        assert card_a.card_id in merged_loaded.predecessor_ids
        assert card_b.card_id in merged_loaded.predecessor_ids
        merge_events = [
            e for e in merged_loaded.lineage
            if e.operation == LineageOperation.MERGE
        ]
        assert len(merge_events) == 1
        assert set(merge_events[0].merged_card_ids) == {
            card_a.card_id, card_b.card_id,
        }

        # Source refs collected from both
        source_ids = {ref.id for ref in merged_loaded.source_refs}
        assert "traj-020" in source_ids
        assert "traj-021" in source_ids

        # Superseded cards point to merged
        a_loaded = store.get_card(card_a.card_id)
        assert a_loaded.superseded_by == merged.card_id
        b_loaded = store.get_card(card_b.card_id)
        assert b_loaded.superseded_by == merged.card_id

        # DuckDB sees all three cards with correct statuses
        for cid, expected_status in [
            (card_a.card_id, "superseded"),
            (card_b.card_id, "superseded"),
            (merged.card_id, "active"),
        ]:
            rows = store.fs.query_cards(f"card_id = '{cid}'")
            assert len(rows) == 1, f"DuckDB missing card {cid}"
            assert rows[0]["status"] == expected_status, (
                f"DuckDB status for {cid}: expected {expected_status}, got {rows[0]['status']}"
            )

    def test_merge_three_cards(self, store):
        """Merging three cards should supersede all three."""
        cards = [
            KnowledgeCard(
                title=f"Activation Function: {name}",
                content=f"The {name} activation function.",
                domain="neural_nets",
            )
            for name in ["ReLU", "GELU", "SiLU"]
        ]
        for c in cards:
            record_creation(c, [], agent="test")
            store.add_card(c)

        merged = KnowledgeCard(
            title="Activation Functions Overview",
            content="Common activation functions: ReLU (max(0,x)), "
            "GELU (Gaussian error), SiLU (x*sigmoid(x)).",
            domain="neural_nets",
        )
        merge_cards(cards, merged, agent="organizer")
        for c in cards:
            store.deactivate_card(c)
        store.add_card(merged)

        for c in cards:
            _assert_fs_status(store, c.card_id, CardStatus.SUPERSEDED)
            _assert_not_in_lance(store, c.card_id)

        _assert_fs_status(store, merged.card_id, CardStatus.ACTIVE)
        _assert_in_lance(store, merged.card_id)

        loaded = store.get_card(merged.card_id)
        assert len(loaded.predecessor_ids) == 3


# ===========================================================================
# 4. Split operations
# ===========================================================================


class TestSplit:
    """Test splitting one card into multiple cards."""

    def test_split_into_two(self, store):
        """Splitting a card: original superseded, children active."""
        original = KnowledgeCard(
            title="Memory Hierarchy",
            content="GPU memory hierarchy: registers (fastest), shared memory, "
            "L1/L2 cache, global memory (slowest). Shared memory is "
            "programmer-managed, acts as a scratchpad.",
            tags=["gpu", "memory", "hierarchy"],
            domain="gpu_architecture",
        )
        record_creation(
            original,
            [SourceReference(id="traj-030", type="trajectory")],
            agent="test",
        )
        store.add_card(original)

        child_a = KnowledgeCard(
            title="GPU Register File",
            content="Registers are the fastest memory on GPU. Each SM has a fixed "
            "register file shared among threads. Register pressure limits occupancy.",
            tags=["gpu", "registers"],
            domain="gpu_architecture",
        )
        child_b = KnowledgeCard(
            title="GPU Shared Memory",
            content="Shared memory is a programmer-managed scratchpad on each SM. "
            "Much faster than global memory. Used for data reuse across threads "
            "in a thread block. Bank conflicts reduce throughput.",
            tags=["gpu", "shared_memory"],
            domain="gpu_architecture",
        )

        split_card(
            original,
            [child_a, child_b],
            child_source_refs=[
                [SourceReference(id="traj-030", type="trajectory")],
                [SourceReference(id="traj-030", type="trajectory")],
            ],
            agent="organizer",
            run_tag="run_split",
        )
        store.deactivate_card(original)
        store.add_card(child_a)
        store.add_card(child_b)

        # Original: superseded on FS, removed from Lance
        _assert_fs_status(store, original.card_id, CardStatus.SUPERSEDED)
        _assert_not_in_lance(store, original.card_id)

        # Children: active on FS, in Lance
        _assert_fs_status(store, child_a.card_id, CardStatus.ACTIVE)
        _assert_fs_status(store, child_b.card_id, CardStatus.ACTIVE)
        _assert_in_lance(store, child_a.card_id)
        _assert_in_lance(store, child_b.card_id)

        # Lineage
        a_loaded = store.get_card(child_a.card_id)
        assert original.card_id in a_loaded.predecessor_ids
        split_events = [
            e for e in a_loaded.lineage
            if e.operation == LineageOperation.SPLIT
        ]
        assert len(split_events) == 1
        assert split_events[0].split_from_card_id == original.card_id

        b_loaded = store.get_card(child_b.card_id)
        assert original.card_id in b_loaded.predecessor_ids

        # Source refs propagated
        a_sources = {ref.id for ref in a_loaded.source_refs}
        assert "traj-030" in a_sources

        # Original superseded_by should reference children
        orig_loaded = store.get_card(original.card_id)
        assert orig_loaded.superseded_by is not None

        # DuckDB sees original as superseded, children as active
        orig_rows = store.fs.query_cards(f"card_id = '{original.card_id}'")
        assert len(orig_rows) == 1
        assert orig_rows[0]["status"] == "superseded"
        for child in [child_a, child_b]:
            child_rows = store.fs.query_cards(f"card_id = '{child.card_id}'")
            assert len(child_rows) == 1
            assert child_rows[0]["status"] == "active"

        # Semantic search: "register" should find child_a, not original
        results = store.search("GPU register pressure occupancy", limit=5)
        result_ids = {r["card_id"] for r in results}
        assert child_a.card_id in result_ids
        assert original.card_id not in result_ids

    def test_split_into_three(self, store):
        """Splitting into three children."""
        original = KnowledgeCard(
            title="GPU Synchronization",
            content="Thread synchronization: __syncthreads() within block, "
            "atomics across blocks, cooperative groups for grid-level sync.",
            domain="gpu_programming",
        )
        record_creation(original, [], agent="test")
        store.add_card(original)

        children = [
            KnowledgeCard(
                title="Block Sync", content="__syncthreads()",
                domain="gpu_programming",
            ),
            KnowledgeCard(
                title="Atomics", content="atomicAdd, atomicCAS",
                domain="gpu_programming",
            ),
            KnowledgeCard(
                title="Cooperative Groups", content="grid.sync()",
                domain="gpu_programming",
            ),
        ]
        split_card(original, children, agent="organizer")
        store.deactivate_card(original)
        for c in children:
            store.add_card(c)

        _assert_fs_status(store, original.card_id, CardStatus.SUPERSEDED)
        _assert_not_in_lance(store, original.card_id)
        for c in children:
            _assert_fs_status(store, c.card_id, CardStatus.ACTIVE)
            _assert_in_lance(store, c.card_id)


# ===========================================================================
# 5. Archive operations
# ===========================================================================


class TestArchive:
    """Test archiving cards."""

    def test_archive_removes_from_search(self, store):
        """Archived card should not appear in semantic search."""
        card = KnowledgeCard(
            title="Deprecated CUDA API",
            content="cudaMallocManaged with hints is deprecated in favor of "
            "explicit memory management with cudaMemcpy.",
            tags=["cuda", "deprecated"],
            domain="cuda",
        )
        record_creation(card, [], agent="test")
        store.add_card(card)
        _assert_in_lance(store, card.card_id)

        archive_card(card, agent="organizer", run_tag="run_archive")
        store.deactivate_card(card)

        # Removed from Lance
        _assert_not_in_lance(store, card.card_id)

        # Still on filesystem as archived
        _assert_fs_status(store, card.card_id, CardStatus.ARCHIVED)
        loaded = store.get_card(card.card_id)
        archive_events = [
            e for e in loaded.lineage
            if e.operation == LineageOperation.ARCHIVE
        ]
        assert len(archive_events) == 1

        # DuckDB sees archived status
        rows = store.fs.query_cards(f"card_id = '{card.card_id}'")
        assert len(rows) == 1
        assert rows[0]["status"] == "archived"

        # Not in default list (active only)
        active = store.list_cards(domain="cuda")
        assert card.card_id not in {c.card_id for c in active}

        # But appears in include_superseded list
        all_cards = store.list_cards(domain="cuda", include_superseded=True)
        assert card.card_id in {c.card_id for c in all_cards}

    def test_archive_does_not_affect_search_for_others(self, store):
        """Archiving one card should not affect other cards' searchability."""
        keep = KnowledgeCard(
            title="CUDA Streams",
            content="CUDA streams enable concurrent kernel execution and "
            "overlapping compute with memory transfers.",
            domain="cuda_runtime",
        )
        remove = KnowledgeCard(
            title="CUDA Streams Legacy",
            content="Old stream API (deprecated).",
            domain="cuda_runtime",
        )
        for c in [keep, remove]:
            record_creation(c, [], agent="test")
            store.add_card(c)

        archive_card(remove, agent="test")
        store.deactivate_card(remove)

        _assert_in_lance(store, keep.card_id)
        _assert_not_in_lance(store, remove.card_id)

        results = store.search("CUDA streams concurrent execution", limit=5)
        result_ids = {r["card_id"] for r in results}
        assert keep.card_id in result_ids
        assert remove.card_id not in result_ids


# ===========================================================================
# 6. Cross-cutting: FS/DuckDB + LanceDB consistency
# ===========================================================================


class TestConsistency:
    """Verify FS and LanceDB stay in sync across all operations."""

    def test_lance_count_matches_active_fs_count(self, store):
        """LanceDB row count should equal number of active cards on FS."""
        cards = [
            KnowledgeCard(title=f"Consistency Card {i}", content=f"Content {i}", domain="test")
            for i in range(5)
        ]
        for c in cards:
            record_creation(c, [], agent="test")
            store.add_card(c)

        initial_lance = store._lance.count()
        initial_fs_active = len(store.list_cards(domain="test"))

        # Archive one
        archive_card(cards[0], agent="test")
        store.deactivate_card(cards[0])

        # Revise one
        revised = KnowledgeCard(title="Consistency Card 1 v2", content="Updated", domain="test")
        revise_card(cards[1], revised, agent="test")
        store.deactivate_card(cards[1])
        store.add_card(revised)

        # Now: 3 original active + 1 revised = 4 active, 2 deactivated
        expected_active = initial_fs_active - 2 + 1  # -archived -superseded +revised
        fs_active = len(store.list_cards(domain="test"))
        assert fs_active == expected_active

        lance_count = store._lance.count()
        expected_lance = initial_lance - 2 + 1  # same arithmetic
        assert lance_count == expected_lance

    def test_duckdb_sees_all_statuses(self, store):
        """DuckDB should see cards of all statuses (active, superseded, archived)."""
        card = KnowledgeCard(
            title="DuckDB Consistency Test",
            content="Test card for DuckDB query.",
            domain="test_duckdb",
        )
        record_creation(card, [], agent="test")
        store.add_card(card)

        # Active
        results = store.fs.query_cards(f"card_id = '{card.card_id}'")
        assert len(results) == 1
        assert results[0]["status"] == "active"

        # Archive
        archive_card(card, agent="test")
        store.deactivate_card(card)

        results = store.fs.query_cards(f"card_id = '{card.card_id}'")
        assert len(results) == 1
        assert results[0]["status"] == "archived"

    def test_search_quality_with_remote_embeddings(self, store):
        """Remote Qwen3-Embedding-8B should produce high-quality semantic search."""
        cards = [
            KnowledgeCard(
                title="Matrix Multiply Tiling",
                content="Tiling matrix multiplication into blocks that fit in shared "
                "memory reduces global memory accesses. Each thread block computes "
                "one output tile by iterating over input tiles.",
                tags=["matmul", "tiling", "shared_memory"],
                domain="gpu_kernels",
            ),
            KnowledgeCard(
                title="Convolution Implementation",
                content="im2col transforms convolution into matrix multiplication. "
                "Winograd reduces arithmetic operations for small filter sizes.",
                tags=["convolution", "im2col", "winograd"],
                domain="gpu_kernels",
            ),
            KnowledgeCard(
                title="Python Decorators",
                content="Decorators are functions that wrap other functions to add "
                "behavior. Use @functools.wraps to preserve metadata.",
                tags=["python", "decorators"],
                domain="python",
            ),
        ]
        for c in cards:
            record_creation(c, [], agent="test")
            store.add_card(c)

        # GPU-related query should rank GPU cards above Python
        results = store.search(
            "how to optimize matrix multiplication on GPU with shared memory",
            limit=3,
        )
        assert len(results) >= 2
        assert results[0]["card_id"] == cards[0].card_id, (
            f"Expected matmul tiling first, got: {results[0]['title']}"
        )
        # Python decorator should not be first
        assert results[0]["domain"] == "gpu_kernels"


# ===========================================================================
# 7. Combined lifecycle
# ===========================================================================


class TestFullLifecycle:
    """Test a realistic card lifecycle: create → revise → merge → split → archive."""

    def test_full_lifecycle(self, store):
        """End-to-end lifecycle across all operations."""
        # Step 1: Create two related cards
        card_a = KnowledgeCard(
            title="Attention: Scaled Dot-Product",
            content="Q K^T / sqrt(d_k) then softmax then multiply by V.",
            tags=["attention", "transformer"],
            domain="ml_kernels",
        )
        card_b = KnowledgeCard(
            title="Attention: Multi-Head",
            content="Split Q/K/V into h heads, apply attention, concat.",
            tags=["attention", "multi_head"],
            domain="ml_kernels",
        )
        for c in [card_a, card_b]:
            record_creation(
                c,
                [SourceReference(id="traj-100", type="trajectory")],
                agent="critic",
            )
            store.add_card(c)

        _assert_lance_count(store, store._lance.count())  # baseline

        # Step 2: Merge into unified attention card
        merged = KnowledgeCard(
            title="Attention Mechanism",
            content="Attention: softmax(QK^T/sqrt(d_k))V. Multi-head splits into h "
            "heads for diverse representation. Flash attention tiles for O(N) memory.",
            tags=["attention", "transformer", "flash_attention"],
            domain="ml_kernels",
        )
        merge_cards([card_a, card_b], merged, agent="organizer")
        store.deactivate_card(card_a)
        store.deactivate_card(card_b)
        store.add_card(merged)

        _assert_fs_status(store, card_a.card_id, CardStatus.SUPERSEDED)
        _assert_fs_status(store, card_b.card_id, CardStatus.SUPERSEDED)
        _assert_fs_status(store, merged.card_id, CardStatus.ACTIVE)

        # Step 3: Revise with more detail
        revised = KnowledgeCard(
            title="Attention Mechanism",
            content="Attention: softmax(QK^T/sqrt(d_k))V. Multi-head: h parallel "
            "attention ops. Flash attention: tiled, online softmax, O(N) memory. "
            "GQA: grouped query attention reduces KV cache.",
            tags=["attention", "flash_attention", "gqa"],
            domain="ml_kernels",
        )
        revise_card(merged, revised, agent="organizer")
        store.deactivate_card(merged)
        store.add_card(revised)

        _assert_fs_status(store, merged.card_id, CardStatus.SUPERSEDED)
        _assert_fs_status(store, revised.card_id, CardStatus.ACTIVE)
        _assert_not_in_lance(store, merged.card_id)
        _assert_in_lance(store, revised.card_id)

        # Step 4: Split into forward and optimization cards
        child_fwd = KnowledgeCard(
            title="Attention Forward Pass",
            content="softmax(QK^T/sqrt(d_k))V. Multi-head and GQA variants.",
            domain="ml_kernels",
        )
        child_opt = KnowledgeCard(
            title="Attention Optimization",
            content="Flash attention: tiled online softmax for O(N) memory. "
            "Paged attention for efficient KV cache.",
            domain="ml_kernels",
        )
        split_card(revised, [child_fwd, child_opt], agent="organizer")
        store.deactivate_card(revised)
        store.add_card(child_fwd)
        store.add_card(child_opt)

        _assert_fs_status(store, revised.card_id, CardStatus.SUPERSEDED)
        _assert_fs_status(store, child_fwd.card_id, CardStatus.ACTIVE)
        _assert_fs_status(store, child_opt.card_id, CardStatus.ACTIVE)

        # Step 5: Archive the forward pass card
        archive_card(child_fwd, agent="organizer")
        store.deactivate_card(child_fwd)

        _assert_fs_status(store, child_fwd.card_id, CardStatus.ARCHIVED)
        _assert_not_in_lance(store, child_fwd.card_id)
        _assert_in_lance(store, child_opt.card_id)

        # Final state: only child_opt is active and searchable
        results = store.search("attention optimization flash", limit=5)
        result_ids = {r["card_id"] for r in results}
        assert child_opt.card_id in result_ids
        # None of the superseded/archived cards should appear
        for old_id in [card_a.card_id, card_b.card_id, merged.card_id,
                       revised.card_id, child_fwd.card_id]:
            assert old_id not in result_ids

        # All cards exist on filesystem (nothing deleted)
        all_ids = [card_a.card_id, card_b.card_id, merged.card_id,
                   revised.card_id, child_fwd.card_id, child_opt.card_id]
        for cid in all_ids:
            assert store.get_card(cid) is not None
