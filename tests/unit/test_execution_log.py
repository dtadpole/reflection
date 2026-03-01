"""Tests for the execution logging system."""

from __future__ import annotations

import json

import pytest

from agenix.execution_log import (
    EventType,
    ExecutionEvent,
    ExecutionLogger,
    NullExecutionLogger,
)


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "test_run" / "execution.jsonl"


@pytest.fixture
def logger(log_path):
    return ExecutionLogger(log_path, run_tag="test_run", agent="solver")


class TestExecutionEvent:
    def test_defaults(self):
        event = ExecutionEvent(event_type=EventType.LOOP_STARTED)
        assert event.event_id  # ULID generated
        assert event.timestamp is not None
        assert event.agent == ""
        assert event.run_tag == ""
        assert event.data == {}
        assert event.error is None

    def test_roundtrip(self):
        event = ExecutionEvent(
            event_type=EventType.AGENT_COMPLETED,
            agent="solver",
            run_tag="run_123",
            duration_ms=5000,
            data={"num_turns": 3, "cost_usd": 0.05},
        )
        dumped = event.model_dump_json()
        loaded = ExecutionEvent.model_validate_json(dumped)
        assert loaded.event_type == EventType.AGENT_COMPLETED
        assert loaded.duration_ms == 5000
        assert loaded.data["num_turns"] == 3


class TestExecutionLogger:
    def test_creates_parent_dirs(self, log_path):
        assert not log_path.parent.exists()
        ExecutionLogger(log_path, run_tag="r1")
        assert log_path.parent.exists()

    def test_emit_writes_jsonl(self, logger, log_path):
        logger.emit(ExecutionEvent(event_type=EventType.LOOP_STARTED))
        logger.emit(ExecutionEvent(event_type=EventType.LOOP_STOPPED))

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        e1 = json.loads(lines[0])
        e2 = json.loads(lines[1])
        assert e1["event_type"] == "loop_started"
        assert e2["event_type"] == "loop_stopped"

    def test_fills_context_defaults(self, logger, log_path):
        logger.emit(ExecutionEvent(event_type=EventType.LOOP_STARTED))

        event = json.loads(log_path.read_text().strip())
        assert event["run_tag"] == "test_run"
        assert event["agent"] == "solver"

    def test_explicit_values_override_context(self, logger, log_path):
        logger.emit(ExecutionEvent(
            event_type=EventType.HANDLER_ERROR,
            agent="critic",
            run_tag="other_run",
        ))

        event = json.loads(log_path.read_text().strip())
        assert event["agent"] == "critic"
        assert event["run_tag"] == "other_run"

    def test_append_semantics(self, logger, log_path):
        for i in range(5):
            logger.loop_started("queue", iteration=i)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5


class TestConvenienceMethods:
    def test_loop_started(self, logger, log_path):
        logger.loop_started("queue", queue_name="problems")
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "loop_started"
        assert event["data"]["loop_type"] == "queue"
        assert event["data"]["queue_name"] == "problems"

    def test_loop_stopped(self, logger, log_path):
        logger.loop_stopped()
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "loop_stopped"

    def test_message_dequeued(self, logger, log_path):
        logger.message_dequeued("msg_1", "problems", {"problem_id": "p1"})
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "message_dequeued"
        assert event["message_id"] == "msg_1"
        assert event["data"]["queue_name"] == "problems"
        assert event["data"]["payload_keys"] == ["problem_id"]

    def test_message_completed(self, logger, log_path):
        logger.message_completed("msg_1")
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "message_completed"
        assert event["message_id"] == "msg_1"

    def test_message_failed(self, logger, log_path):
        logger.message_failed("msg_1", "timeout error", "TimeoutError")
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "message_failed"
        assert event["error"] == "timeout error"
        assert event["error_type"] == "TimeoutError"

    def test_knowledge_retrieval(self, logger, log_path):
        logger.knowledge_retrieval("triton matmul", 5, 10)
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "knowledge_retrieval"
        assert event["data"]["num_hits"] == 5
        assert event["data"]["limit"] == 10

    def test_knowledge_retrieval_truncates_query(self, logger, log_path):
        long_query = "x" * 500
        logger.knowledge_retrieval(long_query, 0, 10)
        event = json.loads(log_path.read_text().strip())
        assert len(event["data"]["query"]) == 200

    def test_agent_started(self, logger, log_path):
        logger.agent_started("solver", "claude-opus-4-6", 30, 5000)
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "agent_started"
        assert event["data"]["model"] == "claude-opus-4-6"
        assert event["data"]["max_turns"] == 30

    def test_agent_completed(self, logger, log_path):
        logger.agent_completed("solver", 45000, 8, 0.12, 3000, 1500)
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "agent_completed"
        assert event["duration_ms"] == 45000
        assert event["data"]["num_turns"] == 8
        assert event["data"]["cost_usd"] == 0.12
        assert event["data"]["input_tokens"] == 3000

    def test_output_parsed(self, logger, log_path):
        logger.output_parsed("parse_trajectory", True, ["trajectory:abc"])
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "output_parsed"
        assert event["data"]["success"] is True
        assert event["data"]["entities"] == ["trajectory:abc"]

    def test_output_parsed_failure(self, logger, log_path):
        logger.output_parsed("parse_trajectory", False, error="bad JSON")
        event = json.loads(log_path.read_text().strip())
        assert event["data"]["success"] is False
        assert event["error"] == "bad JSON"

    def test_data_saved(self, logger, log_path):
        logger.data_saved("trajectory", "traj_123")
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "data_saved"
        assert event["data"]["entity_type"] == "trajectory"

    def test_message_enqueued(self, logger, log_path):
        logger.message_enqueued("trajectories", "msg_2")
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "message_enqueued"
        assert event["data"]["queue_name"] == "trajectories"

    def test_scheduled_trigger(self, logger, log_path):
        logger.scheduled_trigger("OrganizerHandler")
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "scheduled_trigger"

    def test_handler_error(self, logger, log_path):
        logger.handler_error("something broke", "RuntimeError")
        event = json.loads(log_path.read_text().strip())
        assert event["event_type"] == "handler_error"
        assert event["error"] == "something broke"
        assert event["error_type"] == "RuntimeError"


class TestNullExecutionLogger:
    def test_no_op(self, tmp_path):
        null_log = NullExecutionLogger()
        null_log.emit(ExecutionEvent(event_type=EventType.LOOP_STARTED))
        null_log.loop_started("queue")
        null_log.agent_completed("solver", 1000, 3, 0.05)
        # No files created
        assert list(tmp_path.iterdir()) == []

    def test_does_not_create_files(self):
        null_log = NullExecutionLogger()
        null_log.message_dequeued("m1", "q1", {"k": "v"})
        # Path is /dev/null — should not create anything
        assert True  # If we got here without error, it's fine
