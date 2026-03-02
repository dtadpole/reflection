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
from claude_agent_sdk.types import AssistantMessage, SystemMessage, UserMessage

from agenix.conversation_log import ConversationLogger, NullConversationLogger
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
    conversation_path: Optional[Path] = None
    experience_id: Optional[str] = None

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


def _parse_thinking(value: str) -> dict:
    """Parse a thinking config string into a ThinkingConfig dict.

    Accepted formats:
      "adaptive"        → {"type": "adaptive"}
      "disabled"        → {"type": "disabled"}
      "enabled:10000"   → {"type": "enabled", "budget_tokens": 10000}
    """
    if value == "adaptive":
        return {"type": "adaptive"}
    if value == "disabled":
        return {"type": "disabled"}
    if value.startswith("enabled:"):
        budget = int(value.split(":", 1)[1])
        return {"type": "enabled", "budget_tokens": budget}
    raise ValueError(f"Invalid thinking config: {value!r}")


class ClaudeRunner:
    """AgentRunner implementation using claude_agent_sdk.query()."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        cwd: Optional[Path] = None,
        run_dir: Optional[Path] = None,
        experiences_dir: Optional[Path] = None,
    ) -> None:
        self._registry = tool_registry
        self._cwd = cwd
        self._run_dir = run_dir
        self._experiences_dir = experiences_dir

    def run(
        self,
        agent: LoadedAgent,
        input_payload: str,
        *,
        conversation_path: Path | None = None,
    ) -> AgentResult:
        """Run an agent synchronously. Implements the AgentRunner protocol.

        Each agent call gets its own asyncio.run() for clean isolation.
        Conversation is logged as JSONL to *conversation_path* (if given),
        or to a new file under *run_dir*, or not at all.
        """
        return asyncio.run(
            self._run_async(agent, input_payload, conversation_path=conversation_path)
        )

    async def _run_async(
        self,
        agent: LoadedAgent,
        input_payload: str,
        *,
        conversation_path: Path | None = None,
    ) -> AgentResult:
        """Run an agent asynchronously via claude_agent_sdk.query()."""
        conv, experience_id = self._make_conversation_logger(
            agent.name, conversation_path,
        )
        options = self._build_options(agent)

        logger.info(
            "Running agent %s (model=%s, max_turns=%s)",
            agent.name,
            options.model,
            options.max_turns,
        )

        # Log the initial user prompt
        conv.log_user_text(input_payload)

        result_message: ResultMessage | None = None
        turn = 0
        async for message in query(prompt=input_payload, options=options):
            if isinstance(message, AssistantMessage):
                turn += 1
                self._log_assistant_message(agent.name, turn, message)
                conv.log_assistant(message)
            elif isinstance(message, UserMessage):
                self._log_user_message(agent.name, turn, message)
                conv.log_user(message)
            elif isinstance(message, SystemMessage):
                logger.info(
                    "[%s] system: subtype=%s data=%s",
                    agent.name, message.subtype, message.data,
                )
                conv.log_system(message)
            elif isinstance(message, ResultMessage):
                result_message = message
            else:
                logger.warning(
                    "[%s] unknown message type: %s", agent.name, type(message).__name__,
                )

        if result_message is None:
            raise RuntimeError(f"Agent {agent.name} returned no result")

        conv.log_result(result_message)

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
            conversation_path=conv.path,
            experience_id=experience_id,
        )

        logger.info(
            "Agent %s finished in %dms (%d turns, cost=$%.4f)",
            agent.name,
            result.duration_ms,
            result.num_turns,
            result.cost_usd,
        )

        return result

    @staticmethod
    def _log_assistant_message(agent_name: str, turn: int, msg: AssistantMessage) -> None:
        """Log an assistant message with its content blocks."""
        from claude_agent_sdk.types import TextBlock, ThinkingBlock, ToolUseBlock

        if msg.error:
            logger.warning("[%s] turn %d: error=%s", agent_name, turn, msg.error)

        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                logger.info(
                    "[%s] turn %d: tool_use %s (id=%s)",
                    agent_name, turn, block.name, block.id,
                )
            elif isinstance(block, TextBlock):
                preview = block.text[:200].replace("\n", " ")
                logger.info(
                    "[%s] turn %d: text (%d chars): %s",
                    agent_name, turn, len(block.text), preview,
                )
            elif isinstance(block, ThinkingBlock):
                logger.info(
                    "[%s] turn %d: thinking (%d chars)",
                    agent_name, turn, len(block.thinking),
                )
            else:
                logger.info(
                    "[%s] turn %d: block %s",
                    agent_name, turn, type(block).__name__,
                )

    @staticmethod
    def _log_user_message(agent_name: str, turn: int, msg: UserMessage) -> None:
        """Log a user message (typically tool results)."""
        from claude_agent_sdk.types import ToolResultBlock

        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    content_preview = ""
                    if isinstance(block.content, str):
                        content_preview = block.content[:200].replace("\n", " ")
                    elif isinstance(block.content, list):
                        content_preview = f"[{len(block.content)} parts]"
                    status = "error" if block.is_error else "ok"
                    logger.info(
                        "[%s] turn %d: tool_result (%s) for %s: %s",
                        agent_name, turn, status, block.tool_use_id, content_preview,
                    )
                else:
                    logger.info(
                        "[%s] turn %d: user block %s",
                        agent_name, turn, type(block).__name__,
                    )
        elif isinstance(msg.content, str):
            preview = msg.content[:200].replace("\n", " ")
            logger.info(
                "[%s] turn %d: user text (%d chars): %s",
                agent_name, turn, len(msg.content), preview,
            )

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
            thinking=_parse_thinking(agent.config.thinking),
            effort=agent.config.effort,
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

    def _make_conversation_logger(
        self,
        agent_name: str,
        conversation_path: Path | None = None,
    ) -> tuple[ConversationLogger, str | None]:
        """Create a ConversationLogger and return (logger, experience_id).

        Priority: explicit *conversation_path* > experiences_dir > run_dir > no-op.
        """
        from ulid import ULID

        if conversation_path is not None:
            # Derive experience_id from filename (stem)
            eid = conversation_path.stem
            return ConversationLogger(conversation_path), eid
        if self._experiences_dir is not None:
            eid = str(ULID())
            path = self._experiences_dir / agent_name.lower() / f"{eid}.jsonl"
            logger.info("Experience log: %s", path)
            return ConversationLogger(path), eid
        if self._run_dir is not None:
            eid = str(ULID())
            return ConversationLogger(self._run_dir / f"{eid}.jsonl"), eid
        return NullConversationLogger(), None
