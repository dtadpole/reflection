"""Monkey-patch for claude_agent_sdk MCP transport race condition.

Root cause: For string prompts, the SDK calls `end_input()` (closing stdin)
immediately after writing the prompt (client.py:134). But the CLI sends MCP
control requests (tool calls routed to SDK MCP servers) throughout the entire
conversation via stdout, and expects responses via stdin. Closing stdin early
breaks all MCP tool calls.

The SDK already handles this correctly for AsyncIterable prompts — see
`stream_input()` in query.py:582-600 — by waiting for `_first_result_event`
before calling `end_input()`. The bug is that string prompts skip this logic.

Fix: Patch `end_input()` to defer stdin closure until the first result message
is received (signaled via `_first_result_event`). This mirrors the existing
`stream_input()` behavior. We also patch `_handle_control_request` to catch
unrecoverable write errors during cleanup.

Call `apply()` before using claude_agent_sdk.query().
"""

import logging

_applied = False
logger = logging.getLogger(__name__)


def apply() -> None:
    """Apply the monkey-patch to claude_agent_sdk transport and query."""
    global _applied
    if _applied:
        return

    import anyio
    from claude_agent_sdk._internal.query import Query
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    # --- Patch 1: Defer end_input() until result is received ---
    # Save the real implementation for use in the deferred close.
    _original_end_input = SubprocessCLITransport.end_input

    async def _deferred_end_input(self) -> None:
        """Skip immediate close — stdin is closed by _deferred_close_stdin."""
        logger.debug(
            "end_input() deferred — stdin kept open for MCP responses"
        )

    SubprocessCLITransport.end_input = _deferred_end_input  # type: ignore[assignment]

    # --- Patch 2: Schedule deferred stdin close after query.start() ---
    # We patch Query.start() to spawn a background task that waits for the
    # first result event, then closes stdin. This mirrors stream_input().
    _original_start = Query.start

    async def _patched_start(self) -> None:
        await _original_start(self)
        # Only defer if we have SDK MCP servers (otherwise no need)
        if self.sdk_mcp_servers and self._tg:
            self._tg.start_soon(_deferred_close_stdin, self)

    Query.start = _patched_start  # type: ignore[assignment]

    async def _deferred_close_stdin(query_self: Query) -> None:
        """Wait for first result, then close stdin.

        No timeout — this task is cancelled by query.close() when the
        conversation ends (via task group cancellation). A timeout would
        prematurely close stdin while the agent is still working.
        """
        try:
            logger.debug("Waiting for first result before closing stdin")
            await query_self._first_result_event.wait()
            logger.debug("Result received, closing stdin")
            await _original_end_input(query_self.transport)
        except anyio.get_cancelled_exc_class():
            logger.debug("Deferred stdin close cancelled (query ended)")
            raise
        except Exception:
            logger.warning(
                "Error in deferred stdin close", exc_info=True
            )

    # --- Patch 3: Catch unrecoverable errors in _handle_control_request ---
    _original_handle = Query._handle_control_request

    async def _safe_handle_control_request(self, request):
        try:
            await _original_handle(self, request)
        except Exception:
            logger.error(
                "Control request %s failed",
                request.get("request_id", "?"),
                exc_info=True,
            )

    Query._handle_control_request = _safe_handle_control_request  # type: ignore[assignment]

    _applied = True
    logger.debug("Applied claude_agent_sdk MCP fix (deferred end_input)")
