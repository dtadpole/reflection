"""Tests for trajectory helper functions."""

from __future__ import annotations

from agenix.config import StorageConfig
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import Problem, StepType, Trajectory
from agenix.storage.trajectory import (
    add_step,
    complete_trajectory,
    create_trajectory,
    get_trajectory_summary,
    success_rate,
)


class TestTrajectoryHelpers:
    def test_create_trajectory(self):
        t = create_trajectory("prob-123")
        assert t.problem_id == "prob-123"
        assert t.steps == []
        assert t.is_correct is False

    def test_add_step(self):
        t = create_trajectory("prob-123")
        step = add_step(t, StepType.THOUGHT, "Let me think...")
        assert step.step_index == 0
        assert step.step_type == StepType.THOUGHT
        assert len(t.steps) == 1

        step2 = add_step(
            t,
            StepType.ACTION,
            "Running code",
            tool_name="code_executor",
            tool_input="print(42)",
        )
        assert step2.step_index == 1
        assert len(t.steps) == 2

    def test_complete_trajectory(self):
        t = create_trajectory("prob-123")
        add_step(t, StepType.THOUGHT, "thinking")
        complete_trajectory(t, is_correct=True, code_solution="def f(): pass")
        assert t.is_correct is True
        assert t.code_solution == "def f(): pass"
        assert t.completed_at is not None

    def test_get_summary(self):
        t = create_trajectory("prob-123")
        add_step(t, StepType.THOUGHT, "thinking")
        complete_trajectory(t, is_correct=True, code_solution="code")
        summary = get_trajectory_summary(t)
        assert summary["problem_id"] == "prob-123"
        assert summary["is_correct"] is True
        assert summary["num_steps"] == 1
        assert summary["has_code"] is True
        assert summary["completed_at"] is not None

    def test_success_rate(self, tmp_path):
        config = StorageConfig(data_root=str(tmp_path), env="test")
        backend = FSBackend(config)
        backend.initialize()

        p = Problem(title="Test", description="Test problem")
        run = "run_test"

        t1 = Trajectory(problem_id=p.problem_id, is_correct=True)
        t2 = Trajectory(problem_id=p.problem_id, is_correct=False)
        t3 = Trajectory(problem_id=p.problem_id, is_correct=True)
        backend.save_trajectory(t1, run)
        backend.save_trajectory(t2, run)
        backend.save_trajectory(t3, run)

        rate = success_rate(backend, run_tag=run)
        assert abs(rate - 2 / 3) < 0.01

    def test_success_rate_empty(self, tmp_path):
        config = StorageConfig(data_root=str(tmp_path), env="test")
        backend = FSBackend(config)
        backend.initialize()
        assert success_rate(backend) == 0.0
