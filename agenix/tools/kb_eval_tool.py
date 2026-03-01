"""kbEval MCP tool — thin re-export from tools/verifier/kb_eval/logic.py."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import SdkMcpTool

from services.kb_eval.baseline.client import KbEvalClient
from tools.verifier.kb_eval.logic import create_tool


def create_kb_eval_tool(client: KbEvalClient) -> SdkMcpTool[Any]:
    """Create a kb_eval MCP tool backed by the given client.

    Backward-compatible wrapper around tools/verifier/kb_eval/logic.create_tool().
    Note: the tool name is now 'verifier', not 'kb_eval'.
    """
    return create_tool(kb_eval_client=client)


__all__ = ["create_kb_eval_tool"]
