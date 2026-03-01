"""Monkey-patch for claude_agent_sdk MCP transport race condition.

Root cause: After sending the user prompt, the SDK calls `end_input()` which
closes stdin (`_stdin_stream = None`). But MCP control request handlers run
concurrently and may need to write responses back via stdin AFTER end_input()
has been called. The CLI never gets the MCP response (e.g., ListTools) and
marks the MCP server as 'failed'.

Fix: Patch `end_input()` to keep the stdin stream open so MCP control
responses can still be written. The CLI subprocess reads from stdin until
the process exits naturally — closing stdin early is not required.

Also catch unrecoverable write errors in `_handle_control_request` to
prevent ExceptionGroup crashes during cleanup.

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

    from claude_agent_sdk._internal.query import Query
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    # --- Patch 1: Keep stdin open in end_input() ---
    # The original closes _stdin_stream, breaking concurrent MCP writes.
    # Replace with a no-op so the stream stays available for control responses.
    async def _noop_end_input(self) -> None:
        logger.debug("end_input() called — keeping stdin open for MCP control responses")

    SubprocessCLITransport.end_input = _noop_end_input  # type: ignore[assignment]

    # --- Patch 2: Catch unrecoverable errors in _handle_control_request ---
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
    logger.debug("Applied claude_agent_sdk MCP transport fix (end_input noop)")
