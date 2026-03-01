"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def sample_agent_dir(tmp_path: Path) -> Path:
    """Create a sample agent directory: agents/test_agent/base/"""
    variant_dir = tmp_path / "agents" / "test_agent" / "base"
    variant_dir.mkdir(parents=True)

    # agent.md
    (variant_dir / "agent.md").write_text(
        """# Test Agent

## Description
A test agent for unit tests.

## System Prompt
You are a test agent. Follow instructions precisely.

## Input Format
A JSON object with a "task" field.

## Output Format
A JSON object with a "result" field.

## Examples
Input: {"task": "hello"}
Output: {"result": "world"}
"""
    )

    # config.yaml
    (variant_dir / "config.yaml").write_text(
        "model: haiku\ntemperature: 0.5\nmax_turns: 3\n"
        "tools:\n  - Read\ncustom_tools:\n  - code_executor\n"
    )

    # logic.py
    (variant_dir / "logic.py").write_text(
        """def parse_output(text: str) -> dict:
    return {"parsed": text}
"""
    )

    return tmp_path / "agents"


@pytest.fixture
def sample_agent_dir_minimal(tmp_path: Path) -> Path:
    """Create a minimal agent directory: agents/minimal_agent/base/ (only agent.md)."""
    variant_dir = tmp_path / "agents" / "minimal_agent" / "base"
    variant_dir.mkdir(parents=True)

    (variant_dir / "agent.md").write_text(
        """# Minimal Agent

## Description
A minimal agent with defaults.

## System Prompt
You are minimal.
"""
    )

    return tmp_path / "agents"
