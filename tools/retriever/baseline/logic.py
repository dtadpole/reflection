"""Retriever tool (base variant) — LanceDB dense vector retrieval."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.storage.models import CardType
from agenix.tools.base import error_result, text_result
from tools.knowledge.baseline.store import KnowledgeStore


def create_tool(*, knowledge_store: KnowledgeStore) -> SdkMcpTool[Any]:
    """Create a knowledge_retriever MCP tool backed by the given store.

    Returns an SdkMcpTool that can be registered with a ToolRegistry
    or passed directly to create_sdk_mcp_server.
    """

    @tool(
        "knowledge_retriever",
        "Query the knowledge base for relevant cards by semantic similarity",
        {
            "query": str,
            "top_k": int,
            "card_type": str,
        },
    )
    async def knowledge_retriever(args: dict) -> dict:
        query = args.get("query", "")
        if not query:
            return error_result("query parameter is required")

        top_k = args.get("top_k", 5)
        card_type_str = args.get("card_type")

        card_type = None
        if card_type_str:
            try:
                card_type = CardType(card_type_str)
            except ValueError:
                valid = [ct.value for ct in CardType]
                return error_result(
                    f"Invalid card_type '{card_type_str}'. Valid: {valid}"
                )

        results = knowledge_store.search(
            query=query,
            limit=top_k,
            card_type=card_type,
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

    return knowledge_retriever
