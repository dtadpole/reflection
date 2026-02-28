"""Lineage operations for knowledge cards.

Provides helpers to record provenance events (create, revise, merge, split,
archive) and query lineage (reverse lookups, ancestry chains).

The storage system is conceptually immutable: card content is never modified
after creation. Revision, merge, and split all produce NEW cards and supersede
the originals. Cards are never deleted — only superseded or archived.
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


def _sync_source_ids(card: Card) -> None:
    """Keep source_ids in sync with source_refs for backward compat."""
    card.source_ids = [ref.id for ref in card.source_refs]


# --- Mutation helpers ---


def record_creation(
    card: Card,
    source_refs: list[SourceReference],
    agent: str = "",
    run_tag: str = "",
) -> None:
    """Record initial creation of a card from source entities.

    This is the only operation that mutates a card: it sets the initial
    source_refs and appends the CREATE event. After this, the card is
    immutable — any further changes produce new cards.
    """
    card.source_refs = list(source_refs)
    _sync_source_ids(card)
    card.lineage.append(
        LineageEvent(
            operation=LineageOperation.CREATE,
            agent=agent,
            run_tag=run_tag,
            source_refs=list(source_refs),
        )
    )
    card.updated_at = _now()


def revise_card(
    old_card: Card,
    new_card: Card,
    new_source_refs: list[SourceReference] | None = None,
    agent: str = "",
    run_tag: str = "",
) -> None:
    """Revise a card by creating a new card and superseding the old one.

    The new_card should be pre-constructed by the caller with updated content.
    This function:
    - Supersedes the old card (status=SUPERSEDED, superseded_by=new_card)
    - Sets predecessor_ids on the new card
    - Copies + extends source_refs from the old card
    - Appends REVISE event to the new card's lineage
    - Appends SUPERSEDE event to the old card's lineage

    Args:
        old_card: The existing card to be replaced.
        new_card: A freshly created card with updated content.
        new_source_refs: Additional source references for the revision.
        agent: Agent performing the operation.
        run_tag: Run identifier.
    """
    # Supersede the old card
    old_card.status = CardStatus.SUPERSEDED
    old_card.superseded_by = new_card.card_id
    old_card.lineage.append(
        LineageEvent(
            operation=LineageOperation.SUPERSEDE,
            agent=agent,
            run_tag=run_tag,
            superseded_by=new_card.card_id,
            description=f"Revised into {new_card.card_id}",
        )
    )
    old_card.updated_at = _now()

    # Set up the new card
    new_card.predecessor_ids = [old_card.card_id]

    # Inherit all source_refs from old card
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

    _sync_source_ids(new_card)

    new_card.lineage.append(
        LineageEvent(
            operation=LineageOperation.REVISE,
            agent=agent,
            run_tag=run_tag,
            source_refs=added,
        )
    )
    new_card.updated_at = _now()


def merge_cards(
    sources: list[Card],
    new_card: Card,
    agent: str = "",
    run_tag: str = "",
) -> None:
    """Merge source cards into a new card. All sources are superseded.

    The new_card should be pre-constructed by the caller with merged content.
    This function:
    - Supersedes all source cards (status=SUPERSEDED, superseded_by=new_card)
    - Collects source_refs from all sources into the new card
    - Sets predecessor_ids on the new card
    - Appends MERGE event to the new card's lineage
    - Appends SUPERSEDE event to each source card's lineage

    Args:
        sources: Cards to merge (all will be superseded).
        new_card: A freshly created card with the merged content.
        agent: Agent performing the operation.
        run_tag: Run identifier.
    """
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

    _sync_source_ids(new_card)

    new_card.predecessor_ids = list(dict.fromkeys(
        new_card.predecessor_ids + source_ids
    ))

    new_card.lineage.append(
        LineageEvent(
            operation=LineageOperation.MERGE,
            agent=agent,
            run_tag=run_tag,
            merged_card_ids=source_ids,
        )
    )
    new_card.updated_at = _now()

    # Supersede all source cards
    for src in sources:
        src.status = CardStatus.SUPERSEDED
        src.superseded_by = new_card.card_id
        src.lineage.append(
            LineageEvent(
                operation=LineageOperation.SUPERSEDE,
                agent=agent,
                run_tag=run_tag,
                superseded_by=new_card.card_id,
                description=f"Merged into {new_card.card_id}",
            )
        )
        src.updated_at = _now()


def split_card(
    original: Card,
    new_cards: list[Card],
    child_source_refs: list[list[SourceReference]] | None = None,
    agent: str = "",
    run_tag: str = "",
) -> None:
    """Split original card into new_cards. Original is superseded.

    Args:
        child_source_refs: Per-child source references, one list per new_card.
            The caller decides which of the original's sources are relevant to
            each child. If None, no source_refs are copied (caller is
            responsible for setting them separately).
    """
    for i, nc in enumerate(new_cards):
        nc.predecessor_ids = [original.card_id]

        # Set per-child source_refs if provided
        refs: list[SourceReference] = []
        if child_source_refs is not None:
            refs = child_source_refs[i]
            existing_keys = {(r.id, r.type) for r in nc.source_refs}
            for ref in refs:
                if (ref.id, ref.type) not in existing_keys:
                    nc.source_refs.append(SourceReference(id=ref.id, type=ref.type))
                    existing_keys.add((ref.id, ref.type))
            _sync_source_ids(nc)

        nc.lineage.append(
            LineageEvent(
                operation=LineageOperation.SPLIT,
                agent=agent,
                run_tag=run_tag,
                split_from_card_id=original.card_id,
                source_refs=refs,
            )
        )
        nc.updated_at = _now()

    # Supersede original
    new_ids = [nc.card_id for nc in new_cards]
    original.status = CardStatus.SUPERSEDED
    original.lineage.append(
        LineageEvent(
            operation=LineageOperation.SUPERSEDE,
            agent=agent,
            run_tag=run_tag,
            description=f"Split into {', '.join(new_ids)}",
        )
    )
    original.updated_at = _now()


def archive_card(
    card: Card,
    agent: str = "",
    run_tag: str = "",
) -> None:
    """Mark a card as archived."""
    card.status = CardStatus.ARCHIVED
    card.lineage.append(
        LineageEvent(
            operation=LineageOperation.ARCHIVE,
            agent=agent,
            run_tag=run_tag,
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


def get_source_trajectories(card: Card) -> list[str]:
    """Extract trajectory IDs from a card's source_refs."""
    return [ref.id for ref in card.source_refs if ref.type == "trajectory"]


def get_source_reflections(card: Card) -> list[str]:
    """Extract reflection card IDs from a card's source_refs."""
    return [ref.id for ref in card.source_refs if ref.type == "reflection"]


def get_card_ancestry(
    card: Card,
    all_cards: dict[str, Card],
) -> list[str]:
    """Recursively collect all predecessor card IDs (breadth-first)."""
    visited: list[str] = []
    queue = list(card.predecessor_ids)
    while queue:
        cid = queue.pop(0)
        if cid in visited:
            continue
        visited.append(cid)
        ancestor = all_cards.get(cid)
        if ancestor:
            queue.extend(ancestor.predecessor_ids)
    return visited
