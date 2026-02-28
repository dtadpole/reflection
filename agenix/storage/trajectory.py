"""Trajectory convenience helpers built on the filesystem backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import (
    StepType,
    Trajectory,
    TrajectoryStep,
)


def create_trajectory(problem_id: str) -> Trajectory:
    """Create a new trajectory for a problem."""
    return Trajectory(problem_id=problem_id)


def add_step(
    trajectory: Trajectory,
    step_type: StepType,
    content: str,
    tool_name: Optional[str] = None,
    tool_input: Optional[str] = None,
    tool_output: Optional[str] = None,
) -> TrajectoryStep:
    """Append a step to a trajectory and return it."""
    step = TrajectoryStep(
        step_index=len(trajectory.steps),
        step_type=step_type,
        content=content,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
    )
    trajectory.steps.append(step)
    return step


def complete_trajectory(
    trajectory: Trajectory,
    is_correct: bool,
    code_solution: str = "",
    final_answer: str = "",
) -> None:
    """Mark a trajectory as complete."""
    trajectory.is_correct = is_correct
    trajectory.code_solution = code_solution
    trajectory.final_answer = final_answer
    trajectory.completed_at = datetime.now(timezone.utc)


def success_rate(backend: FSBackend, run_tag: Optional[str] = None) -> float:
    """Compute the success rate of trajectories (0.0-1.0). Returns 0.0 if no trajectories."""
    trajectories = backend.list_trajectories(run_tag=run_tag, limit=999999)
    if not trajectories:
        return 0.0
    correct = sum(1 for t in trajectories if t.is_correct)
    return correct / len(trajectories)


def get_trajectory_summary(trajectory: Trajectory) -> dict:
    """Return a summary dict for display purposes."""
    return {
        "trajectory_id": trajectory.trajectory_id,
        "problem_id": trajectory.problem_id,
        "is_correct": trajectory.is_correct,
        "num_steps": len(trajectory.steps),
        "has_code": bool(trajectory.code_solution),
        "created_at": trajectory.created_at.isoformat(),
        "completed_at": (
            trajectory.completed_at.isoformat() if trajectory.completed_at else None
        ),
    }
