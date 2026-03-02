"""Agent runner backed by claude_agent_sdk."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


def _log_verifier_result(agent_name: str, turn: int, block: Any, problem_title: str = "") -> None:
    """Log a verifier tool result with emoji-annotated correctness and performance."""
    tag = f" [{problem_title}]" if problem_title else ""
    if block.is_error:
        logger.info(
            "🔍 [%s] turn %d%s: VERIFICATION ❌ ERROR: %s",
            agent_name, turn, tag,
            (block.content[:300] if isinstance(block.content, str) else str(block.content)[:300]),
        )
        return

    # Parse verifier JSON from content
    raw = ""
    if isinstance(block.content, str):
        raw = block.content
    elif isinstance(block.content, list):
        # MCP tool results may be [{type: "text", text: "..."}]
        for part in block.content:
            if isinstance(part, dict) and part.get("type") == "text":
                raw = part.get("text", "")
                break
            if hasattr(part, "text"):
                raw = part.text
                break

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.info(
            "🔍 [%s] turn %d%s: VERIFICATION result (unparseable): %s",
            agent_name, turn, tag, raw[:300],
        )
        return

    compiled = data.get("compiled", False)
    correct = data.get("correctness", False)
    runtime = data.get("runtime", -1.0)
    stats = data.get("runtime_stats", {})
    # runtime_stats: nested {"generated": {"mean_ms": ...}} or flat {"generated_ms": ...}
    gen_stats = stats.get("generated")
    ref_stats = stats.get("reference")
    gen_ms = gen_stats.get("mean_ms") if isinstance(gen_stats, dict) else stats.get("generated_ms")
    ref_ms = ref_stats.get("mean_ms") if isinstance(ref_stats, dict) else stats.get("reference_ms")
    # Fallback to top-level runtime if generated_ms not available
    if gen_ms is None and runtime > 0:
        gen_ms = runtime

    # Build emoji summary
    compiled_str = "✅ Compiled" if compiled else "❌ Compile failed"
    correct_str = "✅ Correct" if correct else "❌ Incorrect"

    perf_parts: list[str] = []
    if gen_ms is not None and gen_ms > 0:
        perf_parts.append(f"⏱️  {gen_ms:.3f}ms")
    if ref_ms is not None and ref_ms > 0:
        perf_parts.append(f"ref {ref_ms:.3f}ms")
    if gen_ms and ref_ms and gen_ms > 0 and ref_ms > 0:
        speedup = ref_ms / gen_ms
        if speedup >= 1.0:
            perf_parts.append(f"🚀 {speedup:.2f}x speedup")
        else:
            perf_parts.append(f"🐢 {speedup:.2f}x (slower)")

    perf_str = " | ".join(perf_parts) if perf_parts else "no timing"

    logger.info(
        "🔍 [%s] turn %d%s: VERIFICATION — %s | %s | %s",
        agent_name, turn, tag, compiled_str, correct_str, perf_str,
    )


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
        log_name: str | None = None,
    ) -> AgentResult:
        """Run an agent synchronously. Implements the AgentRunner protocol.

        Each agent call gets its own asyncio.run() for clean isolation.
        Conversation is logged as JSONL to *conversation_path* (if given),
        or to a new file under *run_dir*, or not at all.

        *log_name* overrides the display name in log messages (e.g.
        "solver#1") without affecting the experience directory path.
        """
        return asyncio.run(
            self._run_async(
                agent, input_payload,
                conversation_path=conversation_path, log_name=log_name,
            )
        )

    async def _run_async(
        self,
        agent: LoadedAgent,
        input_payload: str,
        *,
        conversation_path: Path | None = None,
        log_name: str | None = None,
    ) -> AgentResult:
        """Run an agent asynchronously via claude_agent_sdk.query()."""
        conv, experience_id = self._make_conversation_logger(
            agent.name, conversation_path,
        )
        options = self._build_options(agent)
        label = log_name or agent.name

        logger.info(
            "Running agent %s (model=%s, max_turns=%s)",
            label,
            options.model,
            options.max_turns,
        )

        # Extract problem title for verification logs (if input is solver JSON)
        problem_title = ""
        try:
            payload = json.loads(input_payload)
            if isinstance(payload, dict):
                problem_title = payload.get("problem", {}).get("title", "")
        except (json.JSONDecodeError, TypeError):
            pass

        # Log the initial user prompt
        conv.log_user_text(input_payload)

        result_message: ResultMessage | None = None
        turn = 0
        max_turns = options.max_turns or 0
        tool_names: dict[str, str] = {}  # tool_use_id → tool_name
        async for message in query(prompt=input_payload, options=options):
            if isinstance(message, AssistantMessage):
                turn += 1
                self._log_assistant_message(label, turn, message)
                self._track_tool_names(message, tool_names)
                conv.log_assistant(message)
                # Enforce max_turns based on assistant message count
                if max_turns and turn >= max_turns:
                    logger.info(
                        "[%s] Reached max_turns=%d, stopping agent.",
                        label, max_turns,
                    )
                    break
            elif isinstance(message, UserMessage):
                self._log_user_message(label, turn, message, tool_names, problem_title)
                conv.log_user(message)
            elif isinstance(message, SystemMessage):
                logger.info(
                    "[%s] system: subtype=%s data=%s",
                    label, message.subtype, message.data,
                )
                conv.log_system(message)
            elif isinstance(message, ResultMessage):
                result_message = message
            else:
                logger.warning(
                    "[%s] unknown message type: %s", label, type(message).__name__,
                )

        if result_message is not None:
            conv.log_result(result_message)

            if result_message.is_error:
                raise RuntimeError(
                    f"Agent {label} returned an error: {result_message.result}"
                )

            usage = result_message.usage or {}
            result = AgentResult(
                output=result_message.result or "",
                duration_ms=result_message.duration_ms,
                num_turns=turn,
                cost_usd=result_message.total_cost_usd or 0.0,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                conversation_path=conv.path,
                experience_id=experience_id,
            )
        else:
            # Stopped early (max_turns reached) — no ResultMessage from SDK
            logger.warning(
                "[%s] No ResultMessage — agent was stopped at turn %d.",
                label, turn,
            )
            result = AgentResult(
                output="",
                num_turns=turn,
                conversation_path=conv.path,
                experience_id=experience_id,
            )

        logger.info(
            "Agent %s finished (%d turns, cost=$%.4f)",
            label,
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
                chars = len(block.text)
                logger.info(
                    "[%s] turn %d: text (%d chars, ~%d tok): %s",
                    agent_name, turn, chars, chars // 4, preview,
                )
            elif isinstance(block, ThinkingBlock):
                chars = len(block.thinking)
                logger.info(
                    "[%s] turn %d: thinking (%d chars, ~%d tok)",
                    agent_name, turn, chars, chars // 4,
                )
            else:
                logger.info(
                    "[%s] turn %d: block %s",
                    agent_name, turn, type(block).__name__,
                )

    @staticmethod
    def _track_tool_names(msg: AssistantMessage, tool_names: dict[str, str]) -> None:
        """Populate tool_use_id → tool_name mapping from assistant tool_use blocks."""
        from claude_agent_sdk.types import ToolUseBlock

        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                tool_names[block.id] = block.name

    @staticmethod
    def _log_user_message(
        agent_name: str,
        turn: int,
        msg: UserMessage,
        tool_names: dict[str, str] | None = None,
        problem_title: str = "",
    ) -> None:
        """Log a user message (typically tool results)."""
        from claude_agent_sdk.types import ToolResultBlock

        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    tool_name = (tool_names or {}).get(block.tool_use_id, "")
                    # Log verifier results with emoji summary
                    if "verifier" in tool_name:
                        _log_verifier_result(agent_name, turn, block, problem_title)
                        continue
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
            max_thinking_tokens=agent.config.max_thinking_tokens,
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
