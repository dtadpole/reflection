"""Agent runner backed by claude_agent_sdk."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from patches.claude_agent_sdk_mcp_fix import apply as _apply_sdk_fix

_apply_sdk_fix()

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from agenix.execution_log import ExecutionLogger, NullExecutionLogger
from agenix.storage.models import LoadedAgent
from agenix.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent invocation with execution metadata."""

    output: str
    duration_ms: int = 0
    num_turns: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

# Short model name -> full model ID
_MODEL_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5",
}


def resolve_model(short_name: str) -> str:
    """Map a short model name to its full model ID.

    Passes through names that are already full model IDs.
    """
    return _MODEL_MAP.get(short_name, short_name)


class ClaudeRunner:
    """AgentRunner implementation using claude_agent_sdk.query()."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        cwd: Optional[Path] = None,
        execution_log: Optional[ExecutionLogger] = None,
    ) -> None:
        self._registry = tool_registry
        self._cwd = cwd
        self._log = execution_log or NullExecutionLogger()

    def run(self, agent: LoadedAgent, input_payload: str) -> AgentResult:
        """Run an agent synchronously. Implements the AgentRunner protocol.

        Each agent call gets its own asyncio.run() for clean isolation.
        """
        return asyncio.run(self._run_async(agent, input_payload))

    async def _run_async(
        self, agent: LoadedAgent, input_payload: str,
    ) -> AgentResult:
        """Run an agent asynchronously via claude_agent_sdk.query()."""
        options = self._build_options(agent)

        logger.info(
            "Running agent %s (model=%s, max_turns=%s)",
            agent.name,
            options.model,
            options.max_turns,
        )

        self._log.agent_started(
            agent_name=agent.name,
            model=options.model,
            max_turns=options.max_turns,
            input_size_chars=len(input_payload),
        )

        result_message: ResultMessage | None = None
        async for message in query(prompt=input_payload, options=options):
            if isinstance(message, ResultMessage):
                result_message = message

        if result_message is None:
            raise RuntimeError(f"Agent {agent.name} returned no result")

        if result_message.is_error:
            raise RuntimeError(
                f"Agent {agent.name} returned an error: {result_message.result}"
            )

        usage = result_message.usage or {}
        result = AgentResult(
            output=result_message.result or "",
            duration_ms=result_message.duration_ms,
            num_turns=result_message.num_turns,
            cost_usd=result_message.total_cost_usd or 0.0,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

        logger.info(
            "Agent %s finished in %dms (%d turns, cost=$%.4f)",
            agent.name,
            result.duration_ms,
            result.num_turns,
            result.cost_usd,
        )

        self._log.agent_completed(
            agent_name=agent.name,
            duration_ms=result.duration_ms,
            num_turns=result.num_turns,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

        return result

    def _build_options(self, agent: LoadedAgent) -> ClaudeAgentOptions:
        """Map a LoadedAgent to ClaudeAgentOptions."""
        builtin_tools = agent.config.tools or []
        custom_tools = agent.config.custom_tools or []

        # Build MCP servers dict — only include tools actually registered
        mcp_servers: dict = {}
        available_custom: list[str] = []
        if custom_tools and self._registry:
            registered = self._registry.list_tools()
            available_custom = [t for t in custom_tools if t in registered]
            if available_custom:
                mcp_servers["reflection"] = self._registry.create_mcp_server(
                    server_name="reflection",
                    tool_names=available_custom,
                )

        # Build allowed_tools list — only add custom tools that have a backing server
        allowed_tools: list[str] = list(builtin_tools)
        for t in available_custom:
            allowed_tools.append(f"mcp__reflection__{t}")

        options = ClaudeAgentOptions(
            model=resolve_model(agent.config.model),
            system_prompt=agent.system_prompt or None,
            max_turns=agent.config.max_turns,
            permission_mode="bypassPermissions",
            # Clear CLAUDECODE so the spawned CLI doesn't refuse to run
            # as a "nested session" when invoked from within Claude Code.
            env={"CLAUDECODE": ""},
        )

        if allowed_tools:
            options.allowed_tools = allowed_tools
        if mcp_servers:
            options.mcp_servers = mcp_servers
        if self._cwd:
            options.cwd = self._cwd

        return options
