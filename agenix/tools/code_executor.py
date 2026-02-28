"""Sandboxed Python code execution tool."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass

from agenix.config import CodeExecutorConfig


@dataclass
class ExecutionResult:
    """Result of a code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


async def execute_python(
    code: str,
    config: CodeExecutorConfig | None = None,
) -> ExecutionResult:
    """Execute Python code in a subprocess with timeout.

    Runs the code in a fresh Python subprocess with no access to the parent
    process's state. Captures stdout/stderr and enforces a timeout.
    """
    if config is None:
        config = CodeExecutorConfig()

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Prevent the child from inheriting env vars that could be sensitive
            env=_safe_env(),
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {config.timeout_seconds}s",
                exit_code=-1,
                timed_out=True,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Truncate output if too large
        max_bytes = config.max_output_bytes
        if len(stdout) > max_bytes:
            stdout = stdout[:max_bytes] + f"\n... (truncated at {max_bytes} bytes)"
        if len(stderr) > max_bytes:
            stderr = stderr[:max_bytes] + f"\n... (truncated at {max_bytes} bytes)"

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode or 0,
        )

    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=f"Failed to execute code: {e}",
            exit_code=-1,
        )


def execute_python_sync(
    code: str,
    config: CodeExecutorConfig | None = None,
) -> ExecutionResult:
    """Synchronous version of execute_python for use outside async contexts."""
    if config is None:
        config = CodeExecutorConfig()

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=config.timeout_seconds,
            env=_safe_env(),
        )

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        max_bytes = config.max_output_bytes
        if len(stdout) > max_bytes:
            stdout = stdout[:max_bytes] + f"\n... (truncated at {max_bytes} bytes)"
        if len(stderr) > max_bytes:
            stderr = stderr[:max_bytes] + f"\n... (truncated at {max_bytes} bytes)"

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr=f"Execution timed out after {config.timeout_seconds}s",
            exit_code=-1,
            timed_out=True,
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=f"Failed to execute code: {e}",
            exit_code=-1,
        )


def _safe_env() -> dict[str, str]:
    """Create a safe environment for the subprocess.

    Strips potentially sensitive environment variables while keeping
    PATH and other essentials needed for Python to run.
    """
    import os

    safe_keys = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "TEMP",
        "TMP",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    }
    return {k: v for k, v in os.environ.items() if k in safe_keys}
