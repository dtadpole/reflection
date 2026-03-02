"""Composite knowledge store: filesystem + LanceDB.

Combines the filesystem backend (structured JSON, DuckDB queries) with
the LanceDB vector index for semantic search over knowledge cards.
"""

from __future__ import annotations

import json
from typing import Optional

from agenix.config import ReflectionConfig
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import (
    Card,
    CardStatus,
)
from tools.knowledge.baseline.embedder import Embedder
from tools.knowledge.baseline.index import LanceIndex


class KnowledgeStore:
    """Unified interface for storing and retrieving knowledge cards.

    - Filesystem stores the full card JSON
    - LanceDB stores the vector embedding for semantic search
    - DuckDB queries are available via FSBackend for metadata filtering
    """

    def __init__(
        self,
        config: ReflectionConfig | None = None,
        fs_backend: FSBackend | None = None,
        lance_index: LanceIndex | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if config is None:
            config = ReflectionConfig()
        self._config = config
        self._fs = fs_backend or FSBackend(config.storage)
        self._embedder = embedder or Embedder(config.embedder)
        self._lance = lance_index or LanceIndex(
            db_path=config.storage.lance_path,
            vector_dim=self._embedder.dimension,
        )

    @property
    def fs(self) -> FSBackend:
        return self._fs

    def initialize(self) -> None:
        """Ensure directories exist."""
        self._fs.initialize()

    def add_card(self, card: Card) -> None:
        """Save a card to filesystem and index its embedding in LanceDB."""
        # Save to filesystem
        self._fs.save_card(card)

        # Embed and index
        text = _card_to_text(card)
        vector = self._embedder.embed_one(text)
        self._lance.add(
            card_id=card.card_id,
            card_type=card.card_type,
            title=card.title,
            domain=card.domain,
            tags=json.dumps(card.tags),
            vector=vector,
        )

    def deactivate_card(self, card: Card) -> None:
        """Persist a non-active card and remove it from the vector index.

        Use this after a lineage operation (revise, merge, split, archive)
        has set the card's status to SUPERSEDED or ARCHIVED. The card is
        saved to the filesystem for lineage tracing but removed from
        LanceDB so it no longer appears in semantic search.
        """
        self._lance.delete(card.card_id)
        self._fs.save_card(card)

    def get_card(self, card_id: str) -> Optional[Card]:
        """Get a card by ID from the filesystem."""
        return self._fs.get_card(card_id)

    def list_cards(
        self,
        card_type: Optional[str] = None,
        domain: Optional[str] = None,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> list[Card]:
        """List cards from the filesystem, optionally filtered.

        By default only returns active cards. Set include_superseded=True
        to also return superseded and archived cards.
        """
        status = None if include_superseded else CardStatus.ACTIVE
        return self._fs.list_cards(
            card_type=card_type, domain=domain, status=status, limit=limit
        )

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        card_type: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search: embed query, search LanceDB, return cards with scores.

        Returns list of dicts with keys: card_id, title, card_type, domain, _distance,
        plus the full card object under 'card'.
        """
        if limit is None:
            limit = self._config.embedder.top_k

        query_vector = self._embedder.embed_one(query)

        # Build optional where clause
        where_parts: list[str] = []
        if card_type:
            where_parts.append(f"card_type = '{card_type}'")
        if domain:
            where_parts.append(f"domain = '{domain}'")
        where = " AND ".join(where_parts) if where_parts else None

        results = self._lance.search(query_vector, limit=limit, where=where)

        # Enrich results with full card data from filesystem
        enriched = []
        for r in results:
            card = self._fs.get_card(r["card_id"])
            if card is not None:
                enriched.append({
                    "card_id": r["card_id"],
                    "title": r.get("title", ""),
                    "card_type": r.get("card_type", ""),
                    "domain": r.get("domain", ""),
                    "_distance": r.get("_distance", 0.0),
                    "card": card,
                })
        return enriched


def _card_to_text(card: Card) -> str:
    """Convert a card to text for embedding."""
    parts = [card.title, card.content]
    if card.tags:
        parts.append("Tags: " + ", ".join(card.tags))
    if card.applicability:
        parts.append("Applicability: " + card.applicability)
    if card.hypothesis:
        parts.append("Hypothesis: " + card.hypothesis)
    return "\n".join(parts)
