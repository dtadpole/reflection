"""Tests for the code executor tool."""

from __future__ import annotations

import pytest

from agenix.config import CodeExecutorConfig
from agenix.tools.code_executor import execute_python, execute_python_sync


class TestCodeExecutorSync:
    def test_simple_output(self):
        result = execute_python_sync("print('hello world')")
        assert result.stdout.strip() == "hello world"
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_stderr(self):
        result = execute_python_sync("import sys; sys.stderr.write('err\\n')")
        assert "err" in result.stderr
        assert result.exit_code == 0

    def test_exit_code(self):
        result = execute_python_sync("import sys; sys.exit(42)")
        assert result.exit_code == 42

    def test_syntax_error(self):
        result = execute_python_sync("def")
        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr

    def test_timeout(self):
        config = CodeExecutorConfig(timeout_seconds=1)
        result = execute_python_sync("import time; time.sleep(10)", config=config)
        assert result.timed_out is True
        assert result.exit_code == -1

    def test_output_truncation(self):
        config = CodeExecutorConfig(max_output_bytes=50)
        result = execute_python_sync("print('x' * 200)", config=config)
        assert len(result.stdout) < 200
        assert "truncated" in result.stdout

    def test_multiline(self):
        code = """
for i in range(3):
    print(i)
"""
        result = execute_python_sync(code)
        assert result.stdout.strip() == "0\n1\n2"

    def test_computation(self):
        code = "print(sum(range(101)))"
        result = execute_python_sync(code)
        assert result.stdout.strip() == "5050"


class TestCodeExecutorAsync:
    @pytest.mark.asyncio
    async def test_simple_output(self):
        result = await execute_python("print('async hello')")
        assert result.stdout.strip() == "async hello"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_timeout(self):
        config = CodeExecutorConfig(timeout_seconds=1)
        result = await execute_python(
            "import time; time.sleep(10)", config=config
        )
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_error(self):
        result = await execute_python("raise ValueError('boom')")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr
