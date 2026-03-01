"""Tests for agent loop abstractions (QueueAgentLoop, ScheduledAgentLoop)."""

from __future__ import annotations

import threading
import time

import pytest

from agenix.agent_loop import QueueAgentLoop, ScheduledAgentLoop
from agenix.config import StorageConfig
from agenix.queue.fs_queue import FSQueue
from agenix.queue.models import MessageState, QueueMessage


@pytest.fixture
def storage_config(tmp_path):
    return StorageConfig(data_root=str(tmp_path), env="test")


@pytest.fixture
def queue(storage_config):
    q = FSQueue("test_queue", storage_config)
    q.initialize()
    return q


class RecordingHandler:
    """A handler that records messages it receives."""

    def __init__(self):
        self.messages: list[QueueMessage] = []

    def handle(self, message: QueueMessage) -> None:
        self.messages.append(message)


class FailingHandler:
    """A handler that raises an exception."""

    def handle(self, message: QueueMessage) -> None:
        raise ValueError("handler error")


class CountingHandler:
    """A scheduled handler that counts invocations."""

    def __init__(self):
        self.count = 0

    def handle(self) -> None:
        self.count += 1


def _run_loop_with_timeout(loop, timeout: float = 0.3):
    """Run a loop in a thread and stop it after timeout."""
    def run_and_stop():
        time.sleep(timeout)
        loop.stop()

    t = threading.Thread(target=run_and_stop)
    t.start()
    loop.run()
    t.join(timeout=timeout + 1.0)


class TestQueueAgentLoop:
    def test_processes_single_message(self, queue):
        handler = RecordingHandler()
        loop = QueueAgentLoop(queue, handler, initial_backoff=0.01, max_backoff=0.01)

        queue.enqueue("test", {"key": "value"})
        _run_loop_with_timeout(loop, timeout=0.2)

        assert len(handler.messages) == 1
        assert handler.messages[0].payload == {"key": "value"}
        assert queue.count(MessageState.DONE) == 1

    def test_processes_multiple_messages_fifo(self, queue):
        handler = RecordingHandler()
        loop = QueueAgentLoop(queue, handler, initial_backoff=0.01, max_backoff=0.01)

        m1 = queue.enqueue("test", {"id": 1})
        m2 = queue.enqueue("test", {"id": 2})
        _run_loop_with_timeout(loop, timeout=0.3)

        assert len(handler.messages) == 2
        assert handler.messages[0].message_id == m1.message_id
        assert handler.messages[1].message_id == m2.message_id
        assert queue.count(MessageState.DONE) == 2

    def test_handler_failure_marks_message_failed(self, queue):
        handler = FailingHandler()
        loop = QueueAgentLoop(queue, handler, initial_backoff=0.01, max_backoff=0.01)

        queue.enqueue("test", {"key": "value"})
        _run_loop_with_timeout(loop, timeout=0.2)

        assert queue.count(MessageState.FAILED) == 1
        assert queue.count(MessageState.DONE) == 0

    def test_empty_queue_does_not_crash(self, queue):
        handler = RecordingHandler()
        loop = QueueAgentLoop(queue, handler, initial_backoff=0.05, max_backoff=0.1)
        _run_loop_with_timeout(loop, timeout=0.2)
        assert len(handler.messages) == 0


class TestScheduledAgentLoop:
    def test_runs_handler_on_interval(self):
        handler = CountingHandler()
        loop = ScheduledAgentLoop(handler, interval=0.05)
        _run_loop_with_timeout(loop, timeout=0.3)

        # Should have run multiple times
        assert handler.count >= 2
        assert handler.count <= 10

    def test_handles_handler_failure(self):
        class FailOnce:
            def __init__(self):
                self.count = 0

            def handle(self):
                self.count += 1
                if self.count == 1:
                    raise ValueError("first run fails")

        handler = FailOnce()
        loop = ScheduledAgentLoop(handler, interval=0.05)
        _run_loop_with_timeout(loop, timeout=0.3)

        # Should continue after failure
        assert handler.count >= 2
