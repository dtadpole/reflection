"""Tests for the code executor MCP tool."""

from __future__ import annotations

import json

import pytest

from agenix.config import CodeExecutorConfig
from agenix.tools.code_executor_tool import create_code_executor_tool


@pytest.fixture
def executor_tool():
    """Create a code_executor tool with default config."""
    return create_code_executor_tool()


@pytest.fixture
def executor_tool_short_timeout():
    """Create a code_executor tool with a short timeout."""
    return create_code_executor_tool(CodeExecutorConfig(timeout_seconds=2))


class TestCreateTool:
    def test_tool_has_correct_name(self, executor_tool):
        assert executor_tool.name == "code_executor"

    def test_tool_creation_with_config(self):
        config = CodeExecutorConfig(timeout_seconds=10, max_output_bytes=1024)
        tool = create_code_executor_tool(config)
        assert tool.name == "code_executor"

    def test_tool_creation_without_config(self):
        tool = create_code_executor_tool()
        assert tool.name == "code_executor"


class TestExecution:
    async def test_simple_print(self, executor_tool):
        result = await executor_tool.handler({"code": "print('hello world')"})
        assert not result.get("is_error")
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["stdout"].strip() == "hello world"
        assert data["exit_code"] == 0
        assert data["timed_out"] is False

    async def test_stderr_output(self, executor_tool):
        result = await executor_tool.handler({"code": "import sys; sys.stderr.write('err\\n')"})
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert "err" in data["stderr"]

    async def test_nonzero_exit_code(self, executor_tool):
        result = await executor_tool.handler({"code": "import sys; sys.exit(1)"})
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["exit_code"] == 1

    async def test_syntax_error(self, executor_tool):
        result = await executor_tool.handler({"code": "def f(:"})
        content = result["content"][0]["text"]
        data = json.loads(content)
        assert data["exit_code"] != 0
        assert "SyntaxError" in data["stderr"]


class TestValidation:
    async def test_empty_code_returns_error(self, executor_tool):
        result = await executor_tool.handler({"code": ""})
        assert result.get("is_error") is True
        assert "required" in result["content"][0]["text"]

    async def test_whitespace_code_returns_error(self, executor_tool):
        result = await executor_tool.handler({"code": "   "})
        assert result.get("is_error") is True

    async def test_missing_code_returns_error(self, executor_tool):
        result = await executor_tool.handler({})
        assert result.get("is_error") is True
