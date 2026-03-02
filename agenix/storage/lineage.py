"""Lineage operations for knowledge cards.

Provides helpers to record provenance events (create, revise, merge, split,
archive) and query lineage.

Cards are never modified after creation. Revision, merge, and split all
produce NEW cards and supersede the originals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from agenix.storage.models import (
    Card,
    CardStatus,
    LineageEvent,
    LineageOperation,
    SourceReference,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- Mutation helpers ---


def record_creation(
    card: Card,
    source_refs: list[SourceReference],
    agent: str = "",
) -> None:
    """Record initial creation of a card from source entities."""
    card.source_refs = list(source_refs)
    card.lineage.append(
        LineageEvent(
            operation=LineageOperation.CREATE,
            agent=agent,
            source_refs=list(source_refs),
        )
    )
    card.updated_at = _now()


def revise_card(
    old_card: Card,
    new_card: Card,
    new_source_refs: list[SourceReference] | None = None,
    agent: str = "",
) -> None:
    """Revise a card: supersede the old, set up the new with lineage."""
    # Supersede the old card
    old_card.status = CardStatus.SUPERSEDED
    old_card.lineage.append(
        LineageEvent(
            operation=LineageOperation.SUPERSEDE,
            agent=agent,
            description=f"Revised into {new_card.card_id}",
        )
    )
    old_card.updated_at = _now()

    # Inherit source_refs from old card
    existing_keys = {(r.id, r.type) for r in new_card.source_refs}
    for ref in old_card.source_refs:
        if (ref.id, ref.type) not in existing_keys:
            new_card.source_refs.append(SourceReference(id=ref.id, type=ref.type))
            existing_keys.add((ref.id, ref.type))

    # Add new source_refs
    added: list[SourceReference] = []
    if new_source_refs:
        for ref in new_source_refs:
            if (ref.id, ref.type) not in existing_keys:
                new_card.source_refs.append(SourceReference(id=ref.id, type=ref.type))
                existing_keys.add((ref.id, ref.type))
                added.append(ref)

    new_card.lineage.append(
        LineageEvent(
            operation=LineageOperation.REVISE,
            agent=agent,
            description=f"Revised from {old_card.card_id}",
            source_refs=added,
        )
    )
    new_card.updated_at = _now()


def merge_cards(
    sources: list[Card],
    new_card: Card,
    agent: str = "",
) -> None:
    """Merge source cards into a new card. All sources are superseded."""
    source_ids = [s.card_id for s in sources]

    # Collect source_refs from all sources
    existing_keys = {(r.id, r.type) for r in new_card.source_refs}
    for src in sources:
        for ref in src.source_refs:
            if (ref.id, ref.type) not in existing_keys:
                new_card.source_refs.append(
                    SourceReference(id=ref.id, type=ref.type)
                )
                existing_keys.add((ref.id, ref.type))

    new_card.lineage.append(
        LineageEvent(
            operation=LineageOperation.MERGE,
            agent=agent,
            description=f"Merged from {', '.join(source_ids)}",
        )
    )
    new_card.updated_at = _now()

    # Supersede all source cards
    for src in sources:
        src.status = CardStatus.SUPERSEDED
        src.lineage.append(
            LineageEvent(
                operation=LineageOperation.SUPERSEDE,
                agent=agent,
                description=f"Merged into {new_card.card_id}",
            )
        )
        src.updated_at = _now()


def split_card(
    original: Card,
    new_cards: list[Card],
    child_source_refs: list[list[SourceReference]] | None = None,
    agent: str = "",
) -> None:
    """Split original card into new_cards. Original is superseded."""
    new_ids = [nc.card_id for nc in new_cards]

    for i, nc in enumerate(new_cards):
        refs: list[SourceReference] = []
        if child_source_refs is not None:
            refs = child_source_refs[i]
            existing_keys = {(r.id, r.type) for r in nc.source_refs}
            for ref in refs:
                if (ref.id, ref.type) not in existing_keys:
                    nc.source_refs.append(SourceReference(id=ref.id, type=ref.type))
                    existing_keys.add((ref.id, ref.type))

        nc.lineage.append(
            LineageEvent(
                operation=LineageOperation.SPLIT,
                agent=agent,
                description=f"Split from {original.card_id}",
                source_refs=refs,
            )
        )
        nc.updated_at = _now()

    # Supersede original
    original.status = CardStatus.SUPERSEDED
    original.lineage.append(
        LineageEvent(
            operation=LineageOperation.SUPERSEDE,
            agent=agent,
            description=f"Split into {', '.join(new_ids)}",
        )
    )
    original.updated_at = _now()


def archive_card(
    card: Card,
    agent: str = "",
) -> None:
    """Mark a card as archived."""
    card.status = CardStatus.ARCHIVED
    card.lineage.append(
        LineageEvent(
            operation=LineageOperation.ARCHIVE,
            agent=agent,
        )
    )
    card.updated_at = _now()


# --- Query helpers ---


def find_cards_by_source(
    source_id: str,
    cards: list[Card],
    source_type: Optional[str] = None,
) -> list[Card]:
    """Find cards referencing a given source entity."""
    results = []
    for card in cards:
        for ref in card.source_refs:
            if ref.id == source_id and (source_type is None or ref.type == source_type):
                results.append(card)
                break
    return results


def get_source_experiences(card: Card) -> list[str]:
    """Extract experience IDs from a card's source_refs."""
    return [ref.id for ref in card.source_refs if ref.type == "experience"]


def get_source_reflections(card: Card) -> list[str]:
    """Extract reflection card IDs from a card's source_refs."""
    return [ref.id for ref in card.source_refs if ref.type == "reflection"]
