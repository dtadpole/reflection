"""Knowledge MCP tools — search, list, get, create, revise, merge, split, archive."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.storage.lineage import (
    archive_card,
    merge_cards,
    record_creation,
    revise_card,
    split_card,
)
from agenix.storage.models import Card, SourceReference
from agenix.tools.base import error_result, text_result
from tools.knowledge.baseline.store import KnowledgeStore


def create_tool(*, knowledge_store: KnowledgeStore) -> list[SdkMcpTool[Any]]:
    """Create knowledge MCP tools backed by the given store.

    Returns a list of SdkMcpTool instances that can be registered
    with a ToolRegistry or passed directly to create_sdk_mcp_server.
    """

    # --- Read tools ---

    @tool(
        "knowledge_search",
        "Semantic search over knowledge cards. Returns cards ranked by relevance.",
        {
            "query": str,
            "top_k": int,
            "card_type": str,
        },
    )
    async def knowledge_search(args: dict) -> dict:
        query = args.get("query", "")
        if not query:
            return error_result("query parameter is required")
        top_k = args.get("top_k", 5)
        results = knowledge_store.search(
            query=query,
            limit=top_k,
            card_type=args.get("card_type") or None,
        )
        cards_out = []
        for r in results:
            card = r["card"]
            cards_out.append({
                "card_id": r["card_id"],
                "title": r["title"],
                "content": card.content,
                "card_type": r["card_type"],
                "tags": card.tags,
                "score": round(1.0 - r.get("_distance", 0.0), 4),
            })
        return text_result(json.dumps({"cards": cards_out}, indent=2))

    @tool(
        "knowledge_list",
        "List knowledge cards, optionally filtered by type.",
        {
            "card_type": str,
            "limit": int,
        },
    )
    async def knowledge_list(args: dict) -> dict:
        cards = knowledge_store.list_cards(
            card_type=args.get("card_type") or None,
            limit=args.get("limit", 100),
        )
        cards_out = [
            {
                "card_id": c.card_id,
                "title": c.title,
                "card_type": c.card_type,
                "tags": c.tags,
            }
            for c in cards
        ]
        return text_result(json.dumps({"cards": cards_out}, indent=2))

    @tool(
        "knowledge_get",
        "Get a single knowledge card by ID with full details.",
        {
            "card_id": str,
        },
    )
    async def knowledge_get(args: dict) -> dict:
        card_id = args.get("card_id", "")
        if not card_id:
            return error_result("card_id parameter is required")
        card = knowledge_store.get_card(card_id)
        if card is None:
            return error_result(f"Card not found: {card_id}")
        return text_result(json.dumps(card.model_dump(), indent=2, default=str))

    # --- Write tools ---

    @tool(
        "knowledge_create",
        "Create a new knowledge card and save it to the knowledge base. "
        "Records lineage and indexes for semantic search.",
        {
            "title": str,
            "content": str,
            "card_type": str,
            "code_snippet": str,
            "tags": str,
            "applicability": str,
            "limitations": str,
            "experience_ids": str,
            "agent": str,
        },
    )
    async def knowledge_create(args: dict) -> dict:
        title = args.get("title", "")
        content = args.get("content", "")
        if not title:
            return error_result("title parameter is required")
        if not content:
            return error_result("content parameter is required")

        # Parse list fields (SDK passes them as JSON strings)
        tags = _parse_list(args.get("tags", ""))
        experience_ids = _parse_list(args.get("experience_ids", ""))

        card = Card(
            card_type=args.get("card_type", "knowledge"),
            title=title,
            content=content,
            code_snippet=args.get("code_snippet", ""),
            tags=tags,
            applicability=args.get("applicability", ""),
            limitations=args.get("limitations", ""),
            experience_ids=experience_ids[:3],
        )

        source_refs = [
            SourceReference(id=eid, type="experience") for eid in experience_ids
        ]
        agent_name = args.get("agent", "")
        record_creation(card, source_refs, agent=agent_name)
        knowledge_store.add_card(card)

        return text_result(json.dumps({
            "card_id": card.card_id,
            "title": card.title,
            "card_type": card.card_type,
        }))

    @tool(
        "knowledge_revise",
        "Revise an existing card. Creates a new card and supersedes the old one.",
        {
            "card_id": str,
            "title": str,
            "content": str,
            "code_snippet": str,
            "tags": str,
            "applicability": str,
            "limitations": str,
            "agent": str,
        },
    )
    async def knowledge_revise(args: dict) -> dict:
        card_id = args.get("card_id", "")
        if not card_id:
            return error_result("card_id parameter is required")

        old_card = knowledge_store.get_card(card_id)
        if old_card is None:
            return error_result(f"Card not found: {card_id}")

        tags = _parse_list(args.get("tags", ""))

        new_card = Card(
            card_type=old_card.card_type,
            title=args.get("title", old_card.title),
            content=args.get("content", old_card.content),
            code_snippet=args.get("code_snippet", old_card.code_snippet),
            tags=tags if tags else old_card.tags,
            applicability=args.get("applicability", old_card.applicability),
            limitations=args.get("limitations", old_card.limitations),
            experience_ids=list(old_card.experience_ids),
        )

        agent_name = args.get("agent", "")
        revise_card(old_card, new_card, agent=agent_name)
        knowledge_store.deactivate_card(old_card)
        knowledge_store.add_card(new_card)

        return text_result(json.dumps({
            "old_card_id": old_card.card_id,
            "new_card_id": new_card.card_id,
            "title": new_card.title,
        }))

    @tool(
        "knowledge_merge",
        "Merge multiple cards into one new card. All source cards are superseded.",
        {
            "card_ids": str,
            "title": str,
            "content": str,
            "code_snippet": str,
            "tags": str,
            "applicability": str,
            "limitations": str,
            "agent": str,
        },
    )
    async def knowledge_merge(args: dict) -> dict:
        card_ids = _parse_list(args.get("card_ids", ""))
        if len(card_ids) < 2:
            return error_result("card_ids must contain at least 2 card IDs")

        title = args.get("title", "")
        content = args.get("content", "")
        if not title:
            return error_result("title parameter is required")
        if not content:
            return error_result("content parameter is required")

        source_cards: list[Card] = []
        for cid in card_ids:
            card = knowledge_store.get_card(cid)
            if card is None:
                return error_result(f"Card not found: {cid}")
            source_cards.append(card)

        tags = _parse_list(args.get("tags", ""))

        # Collect experience_ids from all sources
        all_eids: list[str] = []
        seen: set[str] = set()
        for sc in source_cards:
            for eid in sc.experience_ids:
                if eid not in seen:
                    all_eids.append(eid)
                    seen.add(eid)

        new_card = Card(
            card_type=source_cards[0].card_type,
            title=title,
            content=content,
            code_snippet=args.get("code_snippet", ""),
            tags=tags,
            applicability=args.get("applicability", ""),
            limitations=args.get("limitations", ""),
            experience_ids=all_eids[:3],
        )

        agent_name = args.get("agent", "")
        merge_cards(source_cards, new_card, agent=agent_name)
        for sc in source_cards:
            knowledge_store.deactivate_card(sc)
        knowledge_store.add_card(new_card)

        return text_result(json.dumps({
            "merged_card_ids": card_ids,
            "new_card_id": new_card.card_id,
            "title": new_card.title,
        }))

    @tool(
        "knowledge_split",
        "Split one card into multiple new cards. The original is superseded.",
        {
            "card_id": str,
            "new_cards": str,
            "agent": str,
        },
    )
    async def knowledge_split(args: dict) -> dict:
        card_id = args.get("card_id", "")
        if not card_id:
            return error_result("card_id parameter is required")

        original = knowledge_store.get_card(card_id)
        if original is None:
            return error_result(f"Card not found: {card_id}")

        # new_cards is a JSON array of card specs
        raw = args.get("new_cards", "")
        try:
            specs = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return error_result("new_cards must be a JSON array of card objects")

        if not isinstance(specs, list) or len(specs) < 2:
            return error_result("new_cards must contain at least 2 card specs")

        new_cards: list[Card] = []
        for spec in specs:
            if not isinstance(spec, dict):
                return error_result("Each new_card must be a JSON object")
            tags = spec.get("tags", [])
            if isinstance(tags, str):
                tags = _parse_list(tags)
            new_cards.append(Card(
                card_type=original.card_type,
                title=spec.get("title", ""),
                content=spec.get("content", ""),
                code_snippet=spec.get("code_snippet", ""),
                tags=tags,
                applicability=spec.get("applicability", ""),
                limitations=spec.get("limitations", ""),
                experience_ids=list(original.experience_ids),
            ))

        agent_name = args.get("agent", "")
        split_card(original, new_cards, agent=agent_name)
        knowledge_store.deactivate_card(original)
        for nc in new_cards:
            knowledge_store.add_card(nc)

        return text_result(json.dumps({
            "original_card_id": card_id,
            "new_card_ids": [nc.card_id for nc in new_cards],
        }))

    @tool(
        "knowledge_archive",
        "Archive a card. Removes it from search but preserves it for lineage.",
        {
            "card_id": str,
            "agent": str,
        },
    )
    async def knowledge_archive(args: dict) -> dict:
        card_id = args.get("card_id", "")
        if not card_id:
            return error_result("card_id parameter is required")

        card = knowledge_store.get_card(card_id)
        if card is None:
            return error_result(f"Card not found: {card_id}")

        agent_name = args.get("agent", "")
        archive_card(card, agent=agent_name)
        knowledge_store.deactivate_card(card)

        return text_result(json.dumps({
            "card_id": card_id,
            "status": "archived",
        }))

    return [
        knowledge_search,
        knowledge_list,
        knowledge_get,
        knowledge_create,
        knowledge_revise,
        knowledge_merge,
        knowledge_split,
        knowledge_archive,
    ]


def _parse_list(value: Any) -> list[str]:
    """Parse a list from an agent argument.

    Handles:
    - Already a list: return as-is
    - JSON string: parse the array
    - Comma-separated string: split on commas
    - Empty/falsy: return []
    """
    if isinstance(value, list):
        return [str(v) for v in value]
    if not value:
        return []
    value = str(value).strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
    return [v.strip() for v in value.split(",") if v.strip()]
