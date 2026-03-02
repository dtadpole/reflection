"""Integration test: store cards, embed them, retrieve by semantic search."""

from __future__ import annotations

import pytest

from agenix.config import EmbedderConfig, ReflectionConfig, StorageConfig
from agenix.storage.models import Card
from tools.knowledge.baseline.store import KnowledgeStore


@pytest.fixture
def knowledge_store(tmp_path):
    config = ReflectionConfig(
        storage=StorageConfig(data_root=str(tmp_path), env="test"),
        embedder=EmbedderConfig(model_name="all-MiniLM-L6-v2", top_k=3),
    )
    store = KnowledgeStore(config=config)
    store.initialize()
    return store


class TestKnowledgeRetrieval:
    def test_add_and_search(self, knowledge_store):
        """Store cards, then retrieve by semantic similarity."""
        card1 = Card(card_type="knowledge",
            title="Binary Search",
            content="Binary search divides the search space in half each step. "
            "Requires a sorted array. Time complexity O(log n).",
            tags=["search", "algorithms", "divide-and-conquer"],
            domain="algorithms",
        )
        card2 = Card(card_type="knowledge",
            title="Bubble Sort",
            content="Bubble sort repeatedly swaps adjacent elements. "
            "Simple but O(n^2) time complexity.",
            tags=["sorting", "algorithms"],
            domain="algorithms",
        )
        card3 = Card(card_type="knowledge",
            title="HTTP Status Codes",
            content="200 OK, 404 Not Found, 500 Internal Server Error. "
            "Used in REST API responses.",
            tags=["http", "web", "api"],
            domain="web",
        )

        knowledge_store.add_card(card1)
        knowledge_store.add_card(card2)
        knowledge_store.add_card(card3)

        # Search for algorithm-related content
        results = knowledge_store.search("how to search efficiently in sorted data")
        assert len(results) > 0
        # Binary search should be the most relevant
        assert results[0]["card_id"] == card1.card_id

    def test_search_with_type_filter(self, knowledge_store):
        """Search with card_type filter."""
        k_card = Card(card_type="knowledge",
            title="Recursion",
            content="A function that calls itself to solve smaller subproblems.",
            tags=["recursion"],
            domain="algorithms",
        )
        i_card = Card(card_type="insight",
            title="Recursion Depth",
            content="Python default recursion limit is 1000.",
            tags=["recursion", "python"],
            hypothesis="Increasing recursion limit improves deep recursive solutions",
        )
        knowledge_store.add_card(k_card)
        knowledge_store.add_card(i_card)

        # Search for only insight cards
        results = knowledge_store.search(
            "recursion depth limits", card_type=k_card.card_type
        )
        for r in results:
            assert r["card_type"] == "knowledge"

    def test_search_with_domain_filter(self, knowledge_store):
        """Search with domain filter."""
        card1 = Card(card_type="knowledge",
            title="Graph BFS",
            content="Breadth-first search explores nodes level by level.",
            domain="algorithms",
        )
        card2 = Card(card_type="knowledge",
            title="CSS Flexbox",
            content="Flexbox is a CSS layout model for responsive design.",
            domain="web",
        )
        knowledge_store.add_card(card1)
        knowledge_store.add_card(card2)

        results = knowledge_store.search("layout and design", domain="web")
        assert len(results) > 0
        for r in results:
            assert r["domain"] == "web"

    def test_archived_card_not_in_search(self, knowledge_store):
        """Archived cards should not appear in search results."""
        from agenix.storage.lineage import archive_card

        card = Card(card_type="knowledge",
            title="Linked Lists",
            content="A data structure with nodes pointing to the next node.",
            domain="data_structures",
        )
        knowledge_store.add_card(card)

        results = knowledge_store.search("linked list data structure")
        assert len(results) == 1

        archive_card(card, agent="test")
        knowledge_store.deactivate_card(card)
        results = knowledge_store.search("linked list data structure")
        assert len(results) == 0

        # But card still exists on filesystem
        loaded = knowledge_store.get_card(card.card_id)
        assert loaded is not None
        assert loaded.status.value == "archived"

    def test_revise_card(self, knowledge_store):
        """Revised card creates a new card; old card is archived from search."""
        from agenix.storage.lineage import record_creation, revise_card
        from agenix.storage.models import SourceReference

        old_card = Card(card_type="knowledge",
            title="Hash Tables",
            content="Key-value storage with O(1) average lookup.",
            domain="data_structures",
        )
        record_creation(old_card, [SourceReference(id="traj-1", type="experience")])
        knowledge_store.add_card(old_card)

        # Revise into a new card
        new_card = Card(card_type="knowledge",
            title="Hash Tables",
            content="Hash tables use hash functions for O(1) amortized lookup. "
            "Handle collisions via chaining or open addressing.",
            domain="data_structures",
        )
        revise_card(old_card, new_card)
        knowledge_store.deactivate_card(old_card)
        knowledge_store.add_card(new_card)

        # Old card superseded but still on filesystem
        loaded_old = knowledge_store.get_card(old_card.card_id)
        assert loaded_old is not None
        assert loaded_old.status.value == "superseded"

        # New card is active and searchable
        loaded_new = knowledge_store.get_card(new_card.card_id)
        assert loaded_new is not None
        assert "collisions" in loaded_new.content
