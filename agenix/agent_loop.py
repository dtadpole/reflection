"""Agent loop abstractions for queue-based and scheduled agents.

QueueAgentLoop — Polls an FSQueue, dispatches to a handler, manages message lifecycle.
ScheduledAgentLoop — Runs a handler on a fixed interval.
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Protocol

from agenix.queue.fs_queue import FSQueue
from agenix.queue.models import QueueMessage

logger = logging.getLogger(__name__)


class QueueHandler(Protocol):
    """Handler for queue-based agents."""

    def handle(self, message: QueueMessage) -> None: ...


class ScheduledHandler(Protocol):
    """Handler for timer-based agents."""

    def handle(self) -> None: ...


class QueueAgentLoop:
    """Polls an FSQueue for messages and dispatches them to a handler.

    Features:
    - Exponential backoff on empty queue (1s -> 2s -> 4s -> ... -> max_backoff)
    - Graceful shutdown on SIGINT/SIGTERM
    - Automatic message complete/fail based on handler success
    """

    def __init__(
        self,
        queue: FSQueue,
        handler: QueueHandler,
        *,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> None:
        self._queue = queue
        self._handler = handler
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_factor = backoff_factor
        self._running = False

    def run(self) -> None:
        """Run the loop until shutdown signal."""
        self._running = True
        self._install_signal_handlers()
        self._queue.initialize()

        backoff = self._initial_backoff
        logger.info("Starting queue loop for %s", self._queue.queue_name)

        while self._running:
            message = self._queue.dequeue()
            if message is None:
                time.sleep(backoff)
                backoff = min(backoff * self._backoff_factor, self._max_backoff)
                continue

            # Reset backoff on successful dequeue
            backoff = self._initial_backoff
            self._process(message)

        logger.info("Queue loop for %s stopped", self._queue.queue_name)

    def stop(self) -> None:
        """Signal the loop to stop after the current message."""
        self._running = False

    def _process(self, message: QueueMessage) -> None:
        """Process a single message, completing or failing it."""
        logger.info(
            "Processing message %s from %s",
            message.message_id,
            self._queue.queue_name,
        )
        try:
            self._handler.handle(message)
            self._queue.complete(message.message_id)
            logger.info("Completed message %s", message.message_id)
        except Exception:
            logger.exception("Failed message %s", message.message_id)
            self._queue.fail(message.message_id, error=str(message.message_id))

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        def _shutdown(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received %s, shutting down...", sig_name)
            self.stop()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)


class ScheduledAgentLoop:
    """Runs a handler on a fixed interval.

    Features:
    - Fixed interval between runs (not counting handler execution time)
    - Graceful shutdown on SIGINT/SIGTERM
    - Logs each run
    """

    def __init__(
        self,
        handler: ScheduledHandler,
        *,
        interval: float = 300.0,
    ) -> None:
        self._handler = handler
        self._interval = interval
        self._running = False

    def run(self) -> None:
        """Run the loop until shutdown signal."""
        self._running = True
        self._install_signal_handlers()

        logger.info(
            "Starting scheduled loop (interval=%.0fs)", self._interval
        )

        while self._running:
            try:
                self._handler.handle()
            except Exception:
                logger.exception("Scheduled handler failed")

            # Sleep in small increments so we can respond to shutdown quickly
            elapsed = 0.0
            while self._running and elapsed < self._interval:
                step = min(1.0, self._interval - elapsed)
                time.sleep(step)
                elapsed += step

        logger.info("Scheduled loop stopped")

    def stop(self) -> None:
        """Signal the loop to stop after the current handler run."""
        self._running = False

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        def _shutdown(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received %s, shutting down...", sig_name)
            self.stop()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
