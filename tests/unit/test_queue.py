"""Tests for filesystem-based message queue."""

from __future__ import annotations

import json

import pytest

from agenix.config import StorageConfig
from agenix.queue import FSQueue, MessageState, QueueMessage


@pytest.fixture
def storage_config(tmp_path):
    return StorageConfig(data_root=str(tmp_path), env="test")


@pytest.fixture
def queue(storage_config):
    q = FSQueue("solver", storage_config)
    q.initialize()
    return q


class TestQueueMessage:
    def test_defaults(self):
        msg = QueueMessage(
            queue_name="solver",
            sender="curator",
            payload={"problem_id": "abc123"},
        )
        assert msg.message_id  # ULID generated
        assert msg.queue_name == "solver"
        assert msg.sender == "curator"
        assert msg.payload == {"problem_id": "abc123"}
        assert msg.created_at is not None
        assert msg.claimed_at is None
        assert msg.completed_at is None
        assert msg.error is None

    def test_roundtrip_json(self):
        msg = QueueMessage(
            queue_name="critic",
            sender="solver",
            payload={"trajectory_id": "xyz", "run_tag": "run_001"},
        )
        data = json.loads(msg.model_dump_json())
        restored = QueueMessage.model_validate(data)
        assert restored.message_id == msg.message_id
        assert restored.payload == msg.payload


class TestMessageState:
    def test_values(self):
        assert MessageState.PENDING == "pending"
        assert MessageState.PROCESSING == "processing"
        assert MessageState.DONE == "done"
        assert MessageState.FAILED == "failed"


class TestFSQueueInitialize:
    def test_creates_directories(self, storage_config):
        q = FSQueue("solver", storage_config)
        q.initialize()
        root = storage_config.queues_path / "solver"
        assert (root / "pending").is_dir()
        assert (root / "processing").is_dir()
        assert (root / "done").is_dir()
        assert (root / "failed").is_dir()

    def test_idempotent(self, queue):
        # Second call should not fail
        queue.initialize()


class TestEnqueue:
    def test_creates_pending_file(self, queue, storage_config):
        msg = queue.enqueue("curator", {"problem_id": "p1"})
        path = storage_config.queues_path / "solver" / "pending" / f"{msg.message_id}.json"
        assert path.exists()

    def test_message_content(self, queue):
        msg = queue.enqueue("curator", {"problem_id": "p1"})
        assert msg.queue_name == "solver"
        assert msg.sender == "curator"
        assert msg.payload == {"problem_id": "p1"}

    def test_multiple_enqueue(self, queue):
        m1 = queue.enqueue("curator", {"problem_id": "p1"})
        m2 = queue.enqueue("curator", {"problem_id": "p2"})
        assert m1.message_id != m2.message_id
        assert queue.count(MessageState.PENDING) == 2


class TestDequeue:
    def test_empty_queue_returns_none(self, queue):
        assert queue.dequeue() is None

    def test_claims_message(self, queue):
        enqueued = queue.enqueue("curator", {"problem_id": "p1"})
        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.message_id == enqueued.message_id
        assert dequeued.claimed_at is not None
        assert queue.count(MessageState.PENDING) == 0
        assert queue.count(MessageState.PROCESSING) == 1

    def test_fifo_order(self, queue):
        m1 = queue.enqueue("curator", {"problem_id": "p1"})
        m2 = queue.enqueue("curator", {"problem_id": "p2"})
        d1 = queue.dequeue()
        d2 = queue.dequeue()
        assert d1.message_id == m1.message_id
        assert d2.message_id == m2.message_id

    def test_dequeue_then_empty(self, queue):
        queue.enqueue("curator", {"problem_id": "p1"})
        queue.dequeue()
        assert queue.dequeue() is None


class TestComplete:
    def test_moves_to_done(self, queue):
        msg = queue.enqueue("curator", {"problem_id": "p1"})
        queue.dequeue()
        completed = queue.complete(msg.message_id)
        assert completed.completed_at is not None
        assert queue.count(MessageState.PROCESSING) == 0
        assert queue.count(MessageState.DONE) == 1

    def test_raises_on_missing(self, queue):
        with pytest.raises(OSError):
            queue.complete("nonexistent")


class TestFail:
    def test_moves_to_failed(self, queue):
        msg = queue.enqueue("curator", {"problem_id": "p1"})
        queue.dequeue()
        failed = queue.fail(msg.message_id, "timeout")
        assert failed.completed_at is not None
        assert failed.error == "timeout"
        assert queue.count(MessageState.PROCESSING) == 0
        assert queue.count(MessageState.FAILED) == 1

    def test_raises_on_missing(self, queue):
        with pytest.raises(OSError):
            queue.fail("nonexistent", "err")


class TestPeek:
    def test_empty_returns_none(self, queue):
        assert queue.peek() is None

    def test_reads_without_claiming(self, queue):
        msg = queue.enqueue("curator", {"problem_id": "p1"})
        peeked = queue.peek()
        assert peeked.message_id == msg.message_id
        # Still pending
        assert queue.count(MessageState.PENDING) == 1
        assert queue.count(MessageState.PROCESSING) == 0


class TestListMethods:
    def test_list_pending(self, queue):
        queue.enqueue("curator", {"problem_id": "p1"})
        queue.enqueue("curator", {"problem_id": "p2"})
        msgs = queue.list_pending()
        assert len(msgs) == 2

    def test_list_processing(self, queue):
        queue.enqueue("curator", {"problem_id": "p1"})
        queue.dequeue()
        msgs = queue.list_processing()
        assert len(msgs) == 1

    def test_list_failed(self, queue):
        msg = queue.enqueue("curator", {"problem_id": "p1"})
        queue.dequeue()
        queue.fail(msg.message_id, "err")
        msgs = queue.list_failed()
        assert len(msgs) == 1

    def test_list_with_limit(self, queue):
        for i in range(5):
            queue.enqueue("curator", {"problem_id": f"p{i}"})
        msgs = queue.list_pending(limit=3)
        assert len(msgs) == 3


class TestCount:
    def test_empty(self, queue):
        assert queue.count(MessageState.PENDING) == 0

    def test_after_enqueue(self, queue):
        queue.enqueue("curator", {"problem_id": "p1"})
        assert queue.count(MessageState.PENDING) == 1

    def test_counts_per_state(self, queue):
        m1 = queue.enqueue("curator", {"problem_id": "p1"})
        queue.enqueue("curator", {"problem_id": "p2"})
        queue.dequeue()
        queue.complete(m1.message_id)
        assert queue.count(MessageState.PENDING) == 1
        assert queue.count(MessageState.PROCESSING) == 0
        assert queue.count(MessageState.DONE) == 1


class TestFullLifecycle:
    def test_enqueue_dequeue_complete(self, queue):
        msg = queue.enqueue("curator", {"problem_id": "p1"})
        assert queue.count(MessageState.PENDING) == 1

        dequeued = queue.dequeue()
        assert dequeued.message_id == msg.message_id
        assert queue.count(MessageState.PROCESSING) == 1

        completed = queue.complete(msg.message_id)
        assert completed.completed_at is not None
        assert queue.count(MessageState.DONE) == 1

    def test_enqueue_dequeue_fail(self, queue):
        msg = queue.enqueue("solver", {"trajectory_id": "t1"})
        queue.dequeue()
        failed = queue.fail(msg.message_id, "model error")
        assert failed.error == "model error"
        assert queue.count(MessageState.FAILED) == 1
        assert queue.count(MessageState.PROCESSING) == 0

    def test_multiple_queues_isolated(self, storage_config):
        q1 = FSQueue("solver", storage_config)
        q2 = FSQueue("critic", storage_config)
        q1.initialize()
        q2.initialize()

        q1.enqueue("curator", {"problem_id": "p1"})
        q2.enqueue("solver", {"trajectory_id": "t1"})

        assert q1.count(MessageState.PENDING) == 1
        assert q2.count(MessageState.PENDING) == 1

        d1 = q1.dequeue()
        assert d1.payload == {"problem_id": "p1"}
        assert q2.count(MessageState.PENDING) == 1  # unaffected
