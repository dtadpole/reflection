"""Monkey-patch for claude_agent_sdk MCP transport cleanup bug.

The SDK's `_handle_control_request` method tries to write an error response
back to the CLI subprocess when the initial success write fails with
CLIConnectionError (because the process already exited). The error response
write also fails, and the unhandled exception propagates into the anyio
TaskGroup, causing an ExceptionGroup crash during query cleanup.

Fix: wrap the original method to suppress any exceptions it raises,
since these only occur when the CLI process has already exited.

Call `apply()` before using claude_agent_sdk.query().
"""

import logging

_applied = False
logger = logging.getLogger(__name__)


def apply() -> None:
    """Apply the monkey-patch to claude_agent_sdk._internal.query.Query."""
    global _applied
    if _applied:
        return

    from claude_agent_sdk._internal.query import Query

    _original = Query._handle_control_request

    async def _safe_handle_control_request(self, request):
        try:
            await _original(self, request)
        except Exception:
            # The original handler failed to write a response back to the
            # CLI process (CLIConnectionError). This happens when the process
            # has already exited. Suppressing prevents ExceptionGroup crash
            # during TaskGroup cleanup.
            logger.debug(
                "Suppressed transport error for control request %s",
                request.get("request_id", "?"),
            )

    Query._handle_control_request = _safe_handle_control_request  # type: ignore[assignment]
    _applied = True
    logger.debug("Applied claude_agent_sdk MCP transport fix")
