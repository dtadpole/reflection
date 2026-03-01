"""Tests for agent loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenix.loader import list_agents, list_variants, load_agent, parse_agent_md


class TestParseAgentMd:
    def test_parse_full(self):
        text = """# My Agent

## Description
A helpful agent.

## System Prompt
You are helpful.
Follow instructions.

## Input Format
JSON with a "task" field.

## Output Format
JSON with a "result" field.

## Examples
Input: {"task": "hello"}
Output: {"result": "world"}
"""
        sections = parse_agent_md(text)
        assert sections["name"] == "My Agent"
        assert sections["description"] == "A helpful agent."
        assert "You are helpful." in sections["system_prompt"]
        assert "Follow instructions." in sections["system_prompt"]
        assert "task" in sections["input_format"]
        assert "result" in sections["output_format"]
        assert "hello" in sections["examples"]

    def test_parse_minimal(self):
        text = """# Agent

## Description
Minimal.

## System Prompt
Do things.
"""
        sections = parse_agent_md(text)
        assert sections["name"] == "Agent"
        assert sections["description"] == "Minimal."
        assert sections["system_prompt"] == "Do things."

    def test_multiline_system_prompt(self):
        text = """# Agent

## System Prompt
Line 1.
Line 2.
Line 3.

## Description
After prompt.
"""
        sections = parse_agent_md(text)
        assert "Line 1." in sections["system_prompt"]
        assert "Line 2." in sections["system_prompt"]
        assert "Line 3." in sections["system_prompt"]


class TestLoadAgent:
    def test_load_full_agent(self, sample_agent_dir: Path):
        agent = load_agent("test_agent", agents_dir=sample_agent_dir)
        assert agent.name == "Test Agent"
        assert agent.variant == "base"
        assert agent.description == "A test agent for unit tests."
        assert "test agent" in agent.system_prompt.lower()
        assert agent.config.model == "haiku"
        assert agent.config.temperature == 0.5
        assert agent.config.max_turns == 3
        assert agent.config.tools == ["Read"]
        assert agent.config.custom_tools == ["verifier"]
        assert agent.logic_module_path is not None
        assert agent.logic_module_path.endswith("logic.py")

    def test_load_minimal_agent(self, sample_agent_dir_minimal: Path):
        agent = load_agent("minimal_agent", agents_dir=sample_agent_dir_minimal)
        assert agent.name == "Minimal Agent"
        assert agent.variant == "base"
        assert agent.description == "A minimal agent with defaults."
        assert agent.config.model == "sonnet"
        assert agent.config.temperature == 0.7
        assert agent.logic_module_path is None

    def test_load_specific_variant(self, sample_agent_dir: Path):
        agent = load_agent("test_agent", variant="base", agents_dir=sample_agent_dir)
        assert agent.variant == "base"
        assert agent.name == "Test Agent"

    def test_load_nonexistent_agent(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_agent("nonexistent", agents_dir=agents_dir)

    def test_load_nonexistent_variant(self, sample_agent_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_agent("test_agent", variant="v2", agents_dir=sample_agent_dir)

    def test_load_agent_missing_md(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        variant_dir = agents_dir / "bad_agent" / "base"
        variant_dir.mkdir(parents=True)
        (variant_dir / "config.yaml").write_text("model: haiku\n")
        with pytest.raises(FileNotFoundError):
            load_agent("bad_agent", agents_dir=agents_dir)


class TestListAgents:
    def test_list_agents(self, sample_agent_dir: Path):
        agents = list_agents(agents_dir=sample_agent_dir)
        assert "test_agent" in agents

    def test_list_empty(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        assert list_agents(agents_dir=agents_dir) == []

    def test_list_ignores_non_agent_dirs(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        # Dir without any variant containing agent.md
        (agents_dir / "not_an_agent").mkdir(parents=True)
        # Dir with a variant containing agent.md
        variant_dir = agents_dir / "real_agent" / "base"
        variant_dir.mkdir(parents=True)
        (variant_dir / "agent.md").write_text(
            "# Real Agent\n\n## Description\nReal.\n\n## System Prompt\nHi."
        )
        agents = list_agents(agents_dir=agents_dir)
        assert agents == ["real_agent"]


class TestListVariants:
    def test_list_variants(self, sample_agent_dir: Path):
        variants = list_variants("test_agent", agents_dir=sample_agent_dir)
        assert variants == ["base"]

    def test_list_multiple_variants(self, sample_agent_dir: Path):
        # Add a second variant
        v2_dir = sample_agent_dir / "test_agent" / "v2"
        v2_dir.mkdir()
        (v2_dir / "agent.md").write_text(
            "# Test Agent V2\n\n## Description\nV2.\n\n## System Prompt\nV2."
        )
        variants = list_variants("test_agent", agents_dir=sample_agent_dir)
        assert variants == ["base", "v2"]

    def test_list_variants_nonexistent(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        assert list_variants("nope", agents_dir=agents_dir) == []
