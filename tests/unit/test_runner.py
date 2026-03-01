"""Tests for the agent runner."""

from __future__ import annotations

from agenix.runner import ClaudeRunner, resolve_model
from agenix.storage.models import AgentConfig, LoadedAgent
from agenix.tools.registry import ToolRegistry

# --- resolve_model ---


class TestResolveModel:
    def test_sonnet(self):
        assert resolve_model("sonnet") == "claude-sonnet-4-5"

    def test_opus(self):
        assert resolve_model("opus") == "claude-opus-4-1"

    def test_haiku(self):
        assert resolve_model("haiku") == "claude-haiku-4-5"

    def test_passthrough_full_id(self):
        assert resolve_model("claude-sonnet-4-5") == "claude-sonnet-4-5"

    def test_passthrough_unknown(self):
        assert resolve_model("custom-model-1") == "custom-model-1"


# --- _build_options ---


def _make_agent(**overrides) -> LoadedAgent:
    """Helper to build a LoadedAgent with overrides."""
    defaults = {
        "name": "TestAgent",
        "system_prompt": "You are a test agent.",
        "config": AgentConfig(),
    }
    defaults.update(overrides)
    return LoadedAgent(**defaults)


class TestBuildOptions:
    def test_basic_no_tools(self):
        runner = ClaudeRunner()
        agent = _make_agent()
        opts = runner._build_options(agent)

        assert opts.model == "claude-sonnet-4-5"
        assert opts.system_prompt == "You are a test agent."
        assert opts.max_turns == 10
        assert opts.permission_mode == "bypassPermissions"

    def test_model_mapping(self):
        runner = ClaudeRunner()
        agent = _make_agent(config=AgentConfig(model="opus"))
        opts = runner._build_options(agent)
        assert opts.model == "claude-opus-4-1"

    def test_builtin_tools_only(self):
        runner = ClaudeRunner()
        agent = _make_agent(
            config=AgentConfig(tools=["Read", "Grep", "Glob"]),
        )
        opts = runner._build_options(agent)
        assert "Read" in opts.allowed_tools
        assert "Grep" in opts.allowed_tools
        assert "Glob" in opts.allowed_tools

    def test_custom_tools_only(self):
        registry = ToolRegistry()
        # We don't actually register tools — just verify option building
        runner = ClaudeRunner(tool_registry=registry)
        agent = _make_agent(
            config=AgentConfig(custom_tools=["knowledge_retriever"]),
        )
        opts = runner._build_options(agent)
        assert "mcp__reflection__knowledge_retriever" in opts.allowed_tools

    def test_both_builtin_and_custom(self):
        registry = ToolRegistry()
        runner = ClaudeRunner(tool_registry=registry)
        agent = _make_agent(
            config=AgentConfig(
                tools=["Read"],
                custom_tools=["code_executor"],
            ),
        )
        opts = runner._build_options(agent)
        assert "Read" in opts.allowed_tools
        assert "mcp__reflection__code_executor" in opts.allowed_tools

    def test_empty_system_prompt(self):
        runner = ClaudeRunner()
        agent = _make_agent(system_prompt="")
        opts = runner._build_options(agent)
        assert opts.system_prompt is None

    def test_max_turns_from_config(self):
        runner = ClaudeRunner()
        agent = _make_agent(config=AgentConfig(max_turns=25))
        opts = runner._build_options(agent)
        assert opts.max_turns == 25

    def test_cwd_propagated(self, tmp_path):
        runner = ClaudeRunner(cwd=tmp_path)
        agent = _make_agent()
        opts = runner._build_options(agent)
        assert opts.cwd == tmp_path


class TestClaudeRunnerInit:
    def test_default_init(self):
        runner = ClaudeRunner()
        assert runner._registry is None
        assert runner._cwd is None

    def test_with_registry(self):
        registry = ToolRegistry()
        runner = ClaudeRunner(tool_registry=registry)
        assert runner._registry is registry
