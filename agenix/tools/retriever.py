"""Knowledge retriever MCP tool — thin re-export from tools/retriever/baseline/logic.py."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import SdkMcpTool

from tools.knowledge.baseline.store import KnowledgeStore
from tools.retriever.baseline.logic import create_tool


def create_retriever_tool(store: KnowledgeStore) -> SdkMcpTool[Any]:
    """Create a knowledge_retriever MCP tool backed by the given store.

    Backward-compatible wrapper around tools/retriever/baseline/logic.create_tool().
    """
    return create_tool(knowledge_store=store)


__all__ = ["create_retriever_tool"]
