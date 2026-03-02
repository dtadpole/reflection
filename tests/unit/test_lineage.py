"""Tests for data lineage operations (immutable storage model)."""

from __future__ import annotations

from agenix.storage.lineage import (
    archive_card,
    find_cards_by_source,
    get_card_ancestry,
    get_source_experiences,
    get_source_reflections,
    merge_cards,
    record_creation,
    revise_card,
    split_card,
)
from agenix.storage.models import (
    Card,
    CardStatus,
    LineageOperation,
    SourceReference,
)


def _make_card(**kwargs) -> Card:
    defaults = {"card_type": "knowledge", "title": "Test", "content": "Content"}
    defaults.update(kwargs)
    return Card(**defaults)


class TestRecordCreation:
    def test_sets_source_refs_and_event(self):
        card = _make_card()
        refs = [
            SourceReference(id="traj-1", type="experience"),
            SourceReference(id="refl-1", type="reflection"),
        ]
        record_creation(card, refs, agent="organizer", run_tag="run_001")

        assert len(card.source_refs) == 2
        assert card.source_ids == ["traj-1", "refl-1"]
        assert len(card.lineage) == 1
        event = card.lineage[0]
        assert event.operation == LineageOperation.CREATE
        assert event.agent == "organizer"
        assert event.run_tag == "run_001"
        assert len(event.source_refs) == 2

    def test_status_remains_active(self):
        card = _make_card()
        record_creation(card, [], agent="organizer")
        assert card.status == CardStatus.ACTIVE


class TestReviseCard:
    def test_creates_new_card_and_archives_old(self):
        old = _make_card(title="Old")
        record_creation(
            old,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        new = _make_card(title="Updated")
        revise_card(
            old, new,
            new_source_refs=[SourceReference(id="traj-2", type="experience")],
            agent="organizer",
            run_tag="run_002",
        )

        # Old card is superseded
        assert old.status == CardStatus.SUPERSEDED
        assert old.superseded_by == new.card_id
        assert old.lineage[-1].operation == LineageOperation.SUPERSEDE
        assert old.lineage[-1].superseded_by == new.card_id

        # New card inherits sources + gets new ones
        assert new.predecessor_ids == [old.card_id]
        assert len(new.source_refs) == 2
        assert new.source_ids == ["traj-1", "traj-2"]
        assert new.lineage[-1].operation == LineageOperation.REVISE
        assert len(new.lineage[-1].source_refs) == 1  # only new ref

        # New card is active, old content unchanged
        assert new.status == CardStatus.ACTIVE
        assert old.title == "Old"
        assert new.title == "Updated"

    def test_deduplicates_inherited_sources(self):
        old = _make_card()
        ref = SourceReference(id="traj-1", type="experience")
        record_creation(old, [ref], agent="organizer")

        new = _make_card()
        revise_card(old, new, new_source_refs=[ref], agent="organizer")

        # Only one copy of traj-1
        assert len(new.source_refs) == 1
        assert new.source_ids == ["traj-1"]
        # Event records no new refs (duplicate)
        assert new.lineage[-1].source_refs == []

    def test_revision_without_new_sources(self):
        old = _make_card()
        record_creation(
            old,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        new = _make_card()
        revise_card(old, new, agent="organizer")

        assert old.status == CardStatus.SUPERSEDED
        assert new.predecessor_ids == [old.card_id]
        assert len(new.source_refs) == 1  # inherited from old
        assert new.source_ids == ["traj-1"]

    def test_old_card_content_is_not_modified(self):
        """Immutability: old card's content and source_refs stay unchanged."""
        old = _make_card(title="Original Title", content="Original Content")
        record_creation(
            old,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )
        old_refs_before = list(old.source_refs)

        new = _make_card(title="New Title", content="New Content")
        revise_card(
            old, new,
            new_source_refs=[SourceReference(id="traj-2", type="experience")],
            agent="organizer",
        )

        # Old card's content and source_refs are untouched
        assert old.title == "Original Title"
        assert old.content == "Original Content"
        assert old.source_refs == old_refs_before


class TestMergeCards:
    def test_merge_creates_new_card_and_archives_sources(self):
        card_a = _make_card(title="A")
        record_creation(
            card_a,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        card_b = _make_card(title="B")
        record_creation(
            card_b,
            [SourceReference(id="traj-2", type="experience")],
            agent="organizer",
        )

        merged = _make_card(title="Merged")
        merge_cards([card_a, card_b], merged, agent="organizer", run_tag="run_003")

        # New card gets all sources
        assert len(merged.source_refs) == 2
        assert set(merged.source_ids) == {"traj-1", "traj-2"}
        assert card_a.card_id in merged.predecessor_ids
        assert card_b.card_id in merged.predecessor_ids

        # Merge event on new card
        merge_event = merged.lineage[-1]
        assert merge_event.operation == LineageOperation.MERGE
        assert set(merge_event.merged_card_ids) == {card_a.card_id, card_b.card_id}

        # Sources are superseded
        assert card_a.status == CardStatus.SUPERSEDED
        assert card_a.superseded_by == merged.card_id
        assert card_b.status == CardStatus.SUPERSEDED
        assert card_b.superseded_by == merged.card_id

        # Supersede events on sources
        assert card_a.lineage[-1].operation == LineageOperation.SUPERSEDE
        assert card_b.lineage[-1].operation == LineageOperation.SUPERSEDE

    def test_source_content_unchanged_after_merge(self):
        """Immutability: source cards' content stays untouched."""
        card_a = _make_card(title="A", content="Content A")
        record_creation(card_a, [SourceReference(id="traj-1", type="experience")])

        merged = _make_card(title="Merged")
        merge_cards([card_a], merged, agent="organizer")

        assert card_a.title == "A"
        assert card_a.content == "Content A"


class TestSplitCard:
    def test_split_with_per_child_sources(self):
        """Caller decides which sources go to which child."""
        original = _make_card(title="Original")
        record_creation(
            original,
            [
                SourceReference(id="traj-1", type="experience"),
                SourceReference(id="traj-2", type="experience"),
            ],
            agent="organizer",
        )

        child_a = _make_card(title="Child A")
        child_b = _make_card(title="Child B")

        split_card(
            original,
            [child_a, child_b],
            child_source_refs=[
                [SourceReference(id="traj-1", type="experience")],
                [SourceReference(id="traj-2", type="experience")],
            ],
            agent="organizer",
            run_tag="run_004",
        )

        # Original is superseded
        assert original.status == CardStatus.SUPERSEDED
        assert original.lineage[-1].operation == LineageOperation.SUPERSEDE
        assert child_a.card_id in original.lineage[-1].description
        assert child_b.card_id in original.lineage[-1].description

        # Children reference original
        assert child_a.predecessor_ids == [original.card_id]
        assert child_b.predecessor_ids == [original.card_id]
        assert child_a.lineage[-1].operation == LineageOperation.SPLIT
        assert child_a.lineage[-1].split_from_card_id == original.card_id

        # Each child gets only its relevant sources
        assert get_source_experiences(child_a) == ["traj-1"]
        assert get_source_experiences(child_b) == ["traj-2"]
        assert child_a.source_ids == ["traj-1"]
        assert child_b.source_ids == ["traj-2"]

    def test_split_without_source_refs(self):
        """Without child_source_refs, no sources are copied."""
        original = _make_card(title="Original")
        record_creation(
            original,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        child = _make_card(title="Child")
        split_card(original, [child], agent="organizer")

        assert child.predecessor_ids == [original.card_id]
        assert child.source_refs == []
        assert child.source_ids == []

    def test_original_content_unchanged_after_split(self):
        """Immutability: original card's content stays untouched."""
        original = _make_card(title="Original", content="Original Content")
        record_creation(
            original,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        child = _make_card(title="Child")
        split_card(original, [child], agent="organizer")

        assert original.title == "Original"
        assert original.content == "Original Content"


class TestArchiveCard:
    def test_archive(self):
        card = _make_card()
        archive_card(card, agent="admin", run_tag="run_005")

        assert card.status == CardStatus.ARCHIVED
        assert len(card.lineage) == 1
        assert card.lineage[0].operation == LineageOperation.ARCHIVE
        assert card.lineage[0].agent == "admin"


class TestFindCardsBySource:
    def test_find_by_source_id(self):
        card_a = _make_card(title="A")
        record_creation(
            card_a,
            [SourceReference(id="traj-1", type="experience")],
        )

        card_b = _make_card(title="B")
        record_creation(
            card_b,
            [SourceReference(id="traj-2", type="experience")],
        )

        found = find_cards_by_source("traj-1", [card_a, card_b])
        assert len(found) == 1
        assert found[0].title == "A"

    def test_find_with_type_filter(self):
        card = _make_card()
        record_creation(
            card,
            [
                SourceReference(id="shared-id", type="experience"),
                SourceReference(id="shared-id", type="reflection"),
            ],
        )

        found = find_cards_by_source("shared-id", [card], source_type="experience")
        assert len(found) == 1

    def test_no_match(self):
        card = _make_card()
        record_creation(card, [SourceReference(id="traj-1", type="experience")])
        assert find_cards_by_source("traj-99", [card]) == []


class TestGetSourceHelpers:
    def test_get_source_experiences(self):
        card = _make_card()
        record_creation(
            card,
            [
                SourceReference(id="traj-1", type="experience"),
                SourceReference(id="refl-1", type="reflection"),
                SourceReference(id="traj-2", type="experience"),
            ],
        )
        assert get_source_experiences(card) == ["traj-1", "traj-2"]

    def test_get_source_reflections(self):
        card = _make_card()
        record_creation(
            card,
            [
                SourceReference(id="refl-1", type="reflection"),
                SourceReference(id="traj-1", type="experience"),
            ],
        )
        assert get_source_reflections(card) == ["refl-1"]


class TestGetCardAncestry:
    def test_linear_ancestry(self):
        grandparent = _make_card(title="Grandparent")
        parent = _make_card(title="Parent")
        parent.predecessor_ids = [grandparent.card_id]
        child = _make_card(title="Child")
        child.predecessor_ids = [parent.card_id]

        all_cards = {
            grandparent.card_id: grandparent,
            parent.card_id: parent,
            child.card_id: child,
        }
        ancestry = get_card_ancestry(child, all_cards)
        assert ancestry == [parent.card_id, grandparent.card_id]

    def test_merge_ancestry(self):
        a = _make_card(title="A")
        b = _make_card(title="B")
        merged = _make_card(title="Merged")
        merged.predecessor_ids = [a.card_id, b.card_id]

        all_cards = {a.card_id: a, b.card_id: b, merged.card_id: merged}
        ancestry = get_card_ancestry(merged, all_cards)
        assert set(ancestry) == {a.card_id, b.card_id}

    def test_no_predecessors(self):
        card = _make_card()
        assert get_card_ancestry(card, {card.card_id: card}) == []


class TestBackwardCompat:
    def test_card_without_lineage_fields(self):
        """Cards created without lineage fields should work fine."""
        card = Card(
            card_type="knowledge",
            title="Old Card",
            content="No lineage",
            tags=["legacy"],
        )
        assert card.status == CardStatus.ACTIVE
        assert card.lineage == []
        assert card.source_refs == []
        assert card.superseded_by is None
        assert card.predecessor_ids == []

    def test_json_roundtrip_with_lineage(self):
        card = _make_card()
        record_creation(
            card,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
            run_tag="run_001",
        )

        # Revise into a new card
        new = _make_card(content="Updated")
        revise_card(
            card, new,
            new_source_refs=[SourceReference(id="traj-2", type="experience")],
            agent="organizer",
        )

        # Roundtrip the new card
        json_str = new.model_dump_json()
        restored = Card.model_validate_json(json_str)

        assert len(restored.lineage) == 1
        assert len(restored.source_refs) == 2
        assert restored.source_ids == ["traj-1", "traj-2"]
        assert restored.lineage[0].operation == LineageOperation.REVISE
        assert restored.predecessor_ids == [card.card_id]

        # Roundtrip the superseded card
        json_old = card.model_dump_json()
        restored_old = Card.model_validate_json(json_old)
        assert restored_old.status == CardStatus.SUPERSEDED
        assert restored_old.superseded_by == new.card_id


class TestImmutability:
    """Tests verifying that the storage model is truly immutable."""

    def test_revise_produces_distinct_card_id(self):
        """Revision creates a card with a new card_id, not the same one."""
        old = _make_card()
        record_creation(old, [SourceReference(id="traj-1", type="experience")])

        new = _make_card()
        revise_card(old, new)

        assert old.card_id != new.card_id

    def test_merge_produces_distinct_card_id(self):
        """Merge creates a card with a new card_id."""
        a = _make_card(title="A")
        b = _make_card(title="B")
        merged = _make_card(title="Merged")

        merge_cards([a, b], merged)

        assert merged.card_id != a.card_id
        assert merged.card_id != b.card_id

    def test_all_superseded_cards_have_lineage_events(self):
        """Every superseded card has at least one SUPERSEDE event in its lineage."""
        old = _make_card()
        record_creation(old, [SourceReference(id="traj-1", type="experience")])

        new = _make_card()
        revise_card(old, new, agent="organizer")

        supersede_events = [
            e for e in old.lineage
            if e.operation == LineageOperation.SUPERSEDE
        ]
        assert len(supersede_events) == 1

    def test_revision_chain_preserves_full_lineage(self):
        """Chain of revisions: A → B → C, each is a distinct card."""
        card_a = _make_card(title="V1")
        record_creation(
            card_a,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        card_b = _make_card(title="V2")
        revise_card(
            card_a, card_b,
            new_source_refs=[SourceReference(id="traj-2", type="experience")],
            agent="organizer",
        )

        card_c = _make_card(title="V3")
        revise_card(
            card_b, card_c,
            new_source_refs=[SourceReference(id="traj-3", type="experience")],
            agent="organizer",
        )

        # All three are distinct cards
        assert len({card_a.card_id, card_b.card_id, card_c.card_id}) == 3

        # A and B are archived, C is active
        assert card_a.status == CardStatus.SUPERSEDED
        assert card_b.status == CardStatus.SUPERSEDED
        assert card_c.status == CardStatus.ACTIVE

        # C traces back through the full chain
        all_cards = {
            c.card_id: c for c in [card_a, card_b, card_c]
        }
        ancestry = get_card_ancestry(card_c, all_cards)
        assert card_b.card_id in ancestry
        assert card_a.card_id in ancestry

        # C has all experiences
        assert set(get_source_experiences(card_c)) == {"traj-1", "traj-2", "traj-3"}


class TestTraceability:
    """End-to-end tests: verify cards can trace back to original experiences
    through chains of lineage operations."""

    def test_create_then_revise_preserves_all_experiences(self):
        """Create from traj-1, revise with traj-2 → both traceable on new card."""
        old = _make_card()
        record_creation(
            old,
            [SourceReference(id="traj-1", type="experience")],
            agent="critic",
            run_tag="run_001",
        )

        new = _make_card()
        revise_card(
            old, new,
            new_source_refs=[SourceReference(id="traj-2", type="experience")],
            agent="critic",
            run_tag="run_002",
        )

        trajs = get_source_experiences(new)
        assert trajs == ["traj-1", "traj-2"]

    def test_merge_preserves_all_source_experiences(self):
        """Card A from traj-1, Card B from traj-2, merge → new card traces to both."""
        card_a = _make_card(title="A")
        record_creation(
            card_a,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        card_b = _make_card(title="B")
        record_creation(
            card_b,
            [SourceReference(id="traj-2", type="experience")],
            agent="organizer",
        )

        merged = _make_card(title="Merged")
        merge_cards([card_a, card_b], merged, agent="organizer", run_tag="run_003")

        # Merged card traces back to both experiences
        trajs = get_source_experiences(merged)
        assert set(trajs) == {"traj-1", "traj-2"}

        # Archived cards still retain their own experiences
        assert get_source_experiences(card_a) == ["traj-1"]
        assert get_source_experiences(card_b) == ["traj-2"]

    def test_split_partitions_source_experiences(self):
        """Card from traj-1 + traj-2, split into A(traj-1) + B(traj-2)."""
        original = _make_card(title="Original")
        record_creation(
            original,
            [
                SourceReference(id="traj-1", type="experience"),
                SourceReference(id="traj-2", type="experience"),
            ],
            agent="organizer",
        )

        child_a = _make_card(title="Child A")
        child_b = _make_card(title="Child B")
        split_card(
            original,
            [child_a, child_b],
            child_source_refs=[
                [SourceReference(id="traj-1", type="experience")],
                [SourceReference(id="traj-2", type="experience")],
            ],
            agent="organizer",
        )

        assert get_source_experiences(child_a) == ["traj-1"]
        assert get_source_experiences(child_b) == ["traj-2"]

    def test_merge_then_split_full_chain(self):
        """Create A from traj-1, B from traj-2, merge into C, split C into
        D(traj-1) + E(traj-2). Full ancestry chain is preserved."""
        card_a = _make_card(title="A")
        record_creation(
            card_a,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        card_b = _make_card(title="B")
        record_creation(
            card_b,
            [SourceReference(id="traj-2", type="experience")],
            agent="organizer",
        )

        merged = _make_card(title="Merged")
        merge_cards([card_a, card_b], merged, agent="organizer")

        child_d = _make_card(title="D")
        child_e = _make_card(title="E")
        split_card(
            merged,
            [child_d, child_e],
            child_source_refs=[
                [SourceReference(id="traj-1", type="experience")],
                [SourceReference(id="traj-2", type="experience")],
            ],
            agent="organizer",
        )

        # Each child traces to its relevant experience only
        assert get_source_experiences(child_d) == ["traj-1"]
        assert get_source_experiences(child_e) == ["traj-2"]

        # Ancestry chain: D → merged → A, B
        all_cards = {
            c.card_id: c
            for c in [card_a, card_b, merged, child_d, child_e]
        }
        ancestry_d = get_card_ancestry(child_d, all_cards)
        assert merged.card_id in ancestry_d
        assert card_a.card_id in ancestry_d
        assert card_b.card_id in ancestry_d

    def test_revision_after_merge_adds_new_experience(self):
        """Merge A+B into C, then revise C → new card has all 3 experiences."""
        card_a = _make_card(title="A")
        record_creation(
            card_a,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        card_b = _make_card(title="B")
        record_creation(
            card_b,
            [SourceReference(id="traj-2", type="experience")],
            agent="organizer",
        )

        merged = _make_card(title="Merged")
        merge_cards([card_a, card_b], merged, agent="organizer")

        revised = _make_card(title="Revised")
        revise_card(
            merged, revised,
            new_source_refs=[SourceReference(id="traj-3", type="experience")],
            agent="organizer",
        )

        assert set(get_source_experiences(revised)) == {"traj-1", "traj-2", "traj-3"}
        # merged is superseded, revised is active
        assert merged.status == CardStatus.SUPERSEDED
        assert revised.status == CardStatus.ACTIVE

    def test_find_cards_by_experience_after_operations(self):
        """After create + merge, find_cards_by_source returns the right cards."""
        card_a = _make_card(title="A")
        record_creation(
            card_a,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        card_b = _make_card(title="B")
        record_creation(
            card_b,
            [SourceReference(id="traj-1", type="experience")],
            agent="organizer",
        )

        merged = _make_card(title="Merged")
        merge_cards([card_a, card_b], merged, agent="organizer")

        all_cards = [card_a, card_b, merged]
        found = find_cards_by_source("traj-1", all_cards, source_type="experience")
        # All 3 cards reference traj-1 (A directly, B directly, merged inherited)
        assert len(found) == 3
