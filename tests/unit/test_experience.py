"""Tests for experience helper functions."""

from __future__ import annotations

from agenix.config import StorageConfig
from agenix.storage.experience import (
    add_step,
    complete_experience,
    create_experience,
    get_experience_summary,
    success_rate,
)
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import Experience, Problem, StepType


class TestExperienceHelpers:
    def test_create_experience(self):
        e = create_experience("prob-123")
        assert e.problem_id == "prob-123"
        assert e.steps == []
        assert e.is_correct is False

    def test_add_step(self):
        e = create_experience("prob-123")
        step = add_step(e, StepType.THOUGHT, "Let me think...")
        assert step.step_index == 0
        assert step.step_type == StepType.THOUGHT
        assert len(e.steps) == 1

        step2 = add_step(
            e,
            StepType.ACTION,
            "Running code",
            tool_name="verifier",
            tool_input='{"reference_code": "...", "generated_code": "..."}',
        )
        assert step2.step_index == 1
        assert len(e.steps) == 2

    def test_complete_experience(self):
        e = create_experience("prob-123")
        add_step(e, StepType.THOUGHT, "thinking")
        complete_experience(e, is_correct=True, code_solution="def f(): pass")
        assert e.is_correct is True
        assert e.code_solution == "def f(): pass"
        assert e.completed_at is not None

    def test_get_summary(self):
        e = create_experience("prob-123")
        add_step(e, StepType.THOUGHT, "thinking")
        complete_experience(e, is_correct=True, code_solution="code")
        summary = get_experience_summary(e)
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

        e1 = Experience(problem_id=p.problem_id, is_correct=True)
        e2 = Experience(problem_id=p.problem_id, is_correct=False)
        e3 = Experience(problem_id=p.problem_id, is_correct=True)
        from agenix.storage.fs_backend import _write_json

        for e in (e1, e2, e3):
            path = backend.experiences_dir() / f"{e.experience_id}.json"
            _write_json(path, e)

        rate = success_rate(backend)
        assert abs(rate - 2 / 3) < 0.01

    def test_success_rate_empty(self, tmp_path):
        config = StorageConfig(data_root=str(tmp_path), env="test")
        backend = FSBackend(config)
        backend.initialize()
        assert success_rate(backend) == 0.0
