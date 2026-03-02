"""Experience convenience helpers built on the filesystem backend."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import (
    Experience,
    ExperienceStep,
    StepType,
)


def create_experience(problem_id: str) -> Experience:
    """Create a new experience for a problem."""
    return Experience(problem_id=problem_id)


def add_step(
    experience: Experience,
    step_type: StepType,
    content: str,
    tool_name: Optional[str] = None,
    tool_input: Optional[str] = None,
    tool_output: Optional[str] = None,
) -> ExperienceStep:
    """Append a step to an experience and return it."""
    step = ExperienceStep(
        step_index=len(experience.steps),
        step_type=step_type,
        content=content,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
    )
    experience.steps.append(step)
    return step


def complete_experience(
    experience: Experience,
    is_correct: bool,
    code_solution: str = "",
    final_answer: str = "",
) -> None:
    """Mark an experience as complete."""
    experience.is_correct = is_correct
    experience.code_solution = code_solution
    experience.final_answer = final_answer
    experience.completed_at = datetime.now(timezone.utc)


def success_rate(backend: FSBackend, agent: str = "solver") -> float:
    """Compute the success rate of experiences (0.0-1.0). Returns 0.0 if no experiences."""
    experiences = backend.list_experiences(agent=agent, limit=999999)
    if not experiences:
        return 0.0
    correct = sum(1 for e in experiences if e.is_correct)
    return correct / len(experiences)


def get_experience_summary(experience: Experience) -> dict:
    """Return a summary dict for display purposes."""
    return {
        "experience_id": experience.experience_id,
        "problem_id": experience.problem_id,
        "is_correct": experience.is_correct,
        "num_steps": len(experience.steps),
        "has_code": bool(experience.code_solution),
        "created_at": experience.created_at.isoformat(),
        "completed_at": (
            experience.completed_at.isoformat() if experience.completed_at else None
        ),
    }
