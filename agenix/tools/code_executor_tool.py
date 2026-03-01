"""Code executor MCP tool for running Python code in a sandboxed subprocess."""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from agenix.config import CodeExecutorConfig
from agenix.tools.base import error_result, text_result
from agenix.tools.code_executor import execute_python


def create_code_executor_tool(
    config: CodeExecutorConfig | None = None,
) -> SdkMcpTool[Any]:
    """Create a code_executor MCP tool that runs Python in a subprocess.

    Returns an SdkMcpTool that can be registered with a ToolRegistry
    or passed directly to create_sdk_mcp_server.
    """

    @tool(
        "code_executor",
        "Execute Python code in a sandboxed subprocess and return stdout/stderr",
        {"code": str},
    )
    async def code_executor(args: dict) -> dict:
        code = args.get("code", "")
        if not code.strip():
            return error_result("code parameter is required and must not be empty")

        result = await execute_python(code, config)

        return text_result(json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
        }))

    return code_executor
