"""Retriever tool (rerank variant) — two-stage dense retrieval + cross-encoder reranking."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.tools.base import error_result, text_result
from services.reranker.baseline.client import RerankerClient
from tools.knowledge.baseline.store import KnowledgeStore

_CANDIDATE_MULTIPLIER = 5


def create_tool(
    *, knowledge_store: KnowledgeStore, reranker_client: RerankerClient
) -> SdkMcpTool[Any]:
    """Create a knowledge_retriever MCP tool with two-stage reranking.

    Stage 1: Dense vector retrieval via KnowledgeStore (5*K candidates).
    Stage 2: Cross-encoder reranking via RerankerClient (top K).
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

        card_type = card_type_str or None

        # Stage 1: Dense vector retrieval — fetch 5*K candidates
        candidates = knowledge_store.search(
            query=query,
            limit=top_k * _CANDIDATE_MULTIPLIER,
            card_type=card_type,
        )

        if not candidates:
            return text_result(json.dumps({"cards": []}, indent=2))

        # Stage 2: Cross-encoder reranking
        contents = [r["card"].content for r in candidates]
        rerank_result = await reranker_client.rank(
            query=query, documents=contents
        )

        # Pair scores with candidates, sort by reranker score descending
        scored = list(zip(rerank_result.scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top K
        cards_out = []
        for score, r in scored[:top_k]:
            card = r["card"]
            cards_out.append({
                "card_id": r["card_id"],
                "title": r["title"],
                "content": card.content,
                "card_type": r["card_type"],
                "score": round(score, 4),
            })

        return text_result(json.dumps({"cards": cards_out}, indent=2))

    return knowledge_retriever
