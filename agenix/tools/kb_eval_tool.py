"""kbEval MCP tool for GPU kernel verification."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.tools.base import error_result, text_result
from services.kb_eval.client import KbEvalClient


def create_kb_eval_tool(client: KbEvalClient) -> SdkMcpTool[Any]:
    """Create a kb_eval MCP tool backed by the given client.

    Returns an SdkMcpTool that can be registered with a ToolRegistry
    or passed directly to create_sdk_mcp_server.
    """

    @tool(
        "kb_eval",
        "Evaluate a generated GPU kernel against a reference implementation. "
        "Returns compilation status, correctness, and runtime performance.",
        {
            "reference_code": str,
            "generated_code": str,
            "code_type": str,
        },
    )
    async def kb_eval(args: dict) -> dict:
        reference_code = args.get("reference_code", "")
        generated_code = args.get("generated_code", "")

        if not reference_code.strip():
            return error_result("reference_code parameter is required")
        if not generated_code.strip():
            return error_result("generated_code parameter is required")

        code_type = args.get("code_type", "triton")
        if code_type not in ("triton", "cuda", "pytorch"):
            return error_result(
                f"Invalid code_type '{code_type}'. Must be one of: triton, cuda, pytorch"
            )

        try:
            result = await client.eval(
                reference_code=reference_code,
                generated_code=generated_code,
                code_type=code_type,
            )
            return text_result(json.dumps(result.model_dump(), indent=2))
        except Exception as e:
            return error_result(f"kbEval request failed: {e}")

    return kb_eval
