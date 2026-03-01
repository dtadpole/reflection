"""Knowledge tool (baseline variant) — knowledge base CRUD + semantic search."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.storage.models import CardType
from agenix.tools.base import error_result, text_result
from tools.knowledge.baseline.store import KnowledgeStore


def create_tool(*, knowledge_store: KnowledgeStore) -> SdkMcpTool[Any]:
    """Create a knowledge_store MCP tool backed by the given store.

    Returns an SdkMcpTool that can be registered with a ToolRegistry
    or passed directly to create_sdk_mcp_server.
    """

    @tool(
        "knowledge_store",
        "Manage the knowledge base: search, list, and retrieve cards",
        {
            "action": str,
            "query": str,
            "top_k": int,
            "card_type": str,
            "domain": str,
            "card_id": str,
        },
    )
    async def knowledge_store_tool(args: dict) -> dict:
        action = args.get("action", "")
        if not action:
            return error_result("action parameter is required (search/list/get)")

        if action == "search":
            query = args.get("query", "")
            if not query:
                return error_result("query parameter is required for search")
            top_k = args.get("top_k", 5)
            card_type = _parse_card_type(args.get("card_type"))
            if isinstance(card_type, dict):
                return card_type  # error_result

            results = knowledge_store.search(
                query=query, limit=top_k, card_type=card_type,
                domain=args.get("domain"),
            )
            cards_out = []
            for r in results:
                card = r["card"]
                cards_out.append({
                    "card_id": r["card_id"],
                    "title": r["title"],
                    "content": card.content,
                    "card_type": r["card_type"],
                    "score": round(1.0 - r.get("_distance", 0.0), 4),
                })
            return text_result(json.dumps({"cards": cards_out}, indent=2))

        elif action == "list":
            card_type = _parse_card_type(args.get("card_type"))
            if isinstance(card_type, dict):
                return card_type
            cards = knowledge_store.list_cards(
                card_type=card_type, domain=args.get("domain"),
            )
            cards_out = [
                {"card_id": c.card_id, "title": c.title, "card_type": c.card_type.value}
                for c in cards
            ]
            return text_result(json.dumps({"cards": cards_out}, indent=2))

        elif action == "get":
            card_id = args.get("card_id", "")
            if not card_id:
                return error_result("card_id parameter is required for get")
            card = knowledge_store.get_card(card_id)
            if card is None:
                return error_result(f"Card not found: {card_id}")
            return text_result(json.dumps(card.model_dump(), indent=2, default=str))

        else:
            return error_result(
                f"Unknown action '{action}'. Valid: search, list, get"
            )

    return knowledge_store_tool


def _parse_card_type(card_type_str: str | None) -> CardType | None | dict:
    """Parse card_type string, returning None, CardType, or error_result dict."""
    if not card_type_str:
        return None
    try:
        return CardType(card_type_str)
    except ValueError:
        valid = [ct.value for ct in CardType]
        return error_result(f"Invalid card_type '{card_type_str}'. Valid: {valid}")
