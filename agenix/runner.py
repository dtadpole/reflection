"""Agent runner backed by claude_agent_sdk."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from patches.claude_agent_sdk_mcp_fix import apply as _apply_sdk_fix

_apply_sdk_fix()

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from agenix.storage.models import LoadedAgent
from agenix.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

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
    ) -> None:
        self._registry = tool_registry
        self._cwd = cwd

    def run(self, agent: LoadedAgent, input_payload: str) -> str:
        """Run an agent synchronously. Implements the AgentRunner protocol.

        Each agent call gets its own asyncio.run() for clean isolation.
        """
        return asyncio.run(self._run_async(agent, input_payload))

    async def _run_async(self, agent: LoadedAgent, input_payload: str) -> str:
        """Run an agent asynchronously via claude_agent_sdk.query()."""
        options = self._build_options(agent)

        logger.info(
            "Running agent %s (model=%s, max_turns=%s)",
            agent.name,
            options.model,
            options.max_turns,
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

        logger.info(
            "Agent %s finished in %dms (%d turns, cost=$%.4f)",
            agent.name,
            result_message.duration_ms,
            result_message.num_turns,
            result_message.total_cost_usd or 0,
        )

        return result_message.result or ""

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
        )

        if allowed_tools:
            options.allowed_tools = allowed_tools
        if mcp_servers:
            options.mcp_servers = mcp_servers
        if self._cwd:
            options.cwd = self._cwd

        return options
