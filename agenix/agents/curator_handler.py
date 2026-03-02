"""Curator handler — pure Python KernelBench loader (no LLM).

Loads GPU kernel problems from the ScalingIntelligence/KernelBench
HuggingFace dataset, converts them to Problem models, saves to
FSBackend, and enqueues to the problems queue.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agenix.queue.fs_queue import FSQueue, MessageState
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import Difficulty, Problem

logger = logging.getLogger(__name__)

DATASET_NAME = "ScalingIntelligence/KernelBench"

LEVEL_DIFFICULTY = {
    "level_1": Difficulty.EASY,
    "level_2": Difficulty.MEDIUM,
    "level_3": Difficulty.HARD,
    "level_4": Difficulty.HARD,
}


def load_kernelbench(
    levels: Optional[list[str]] = None,
) -> list[dict]:
    """Load KernelBench dataset from HuggingFace.

    Returns list of dicts with keys: code, level, name, problem_id.
    """
    from datasets import load_dataset

    all_levels = levels or ["level_1", "level_2", "level_3", "level_4"]
    rows = []
    for level in all_levels:
        ds = load_dataset(DATASET_NAME, split=level)
        for row in ds:
            rows.append({
                "code": row["code"],
                "level": level,
                "name": row["name"],
                "kb_problem_id": row.get("problem_id", row["name"]),
            })
    logger.info("Loaded %d problems from KernelBench (levels=%s)", len(rows), all_levels)
    return rows


def sample_problems(
    rows: list[dict],
    n: int,
    seed: Optional[int] = None,
) -> list[dict]:
    """Randomly sample N problems from the loaded rows."""
    rng = random.Random(seed)
    if n >= len(rows):
        return rows
    return rng.sample(rows, n)


def row_to_problem(row: dict) -> Problem:
    """Convert a KernelBench row to a Problem model."""
    level = row["level"]
    name = row["name"]
    code = row["code"]

    return Problem(
        title=f"[KernelBench/{level}] {name}",
        description=(
            f"Convert the following PyTorch code to an optimized Triton GPU kernel.\n\n"
            f"**Level**: {level}\n"
            f"**Problem**: {name}\n\n"
            f"## Reference PyTorch Code\n\n"
            f"```python\n{code}\n```\n\n"
            f"Write a `ModelNew` class that produces the same outputs as the reference "
            f"`Model` class but uses custom Triton kernels for the compute-intensive "
            f"operations. Your implementation must:\n"
            f"1. Be functionally correct (torch.allclose with the reference)\n"
            f"2. Maintain the same interface (same inputs/outputs)\n"
            f"3. Aim for better performance than the PyTorch reference"
        ),
        reference_code=code,
        domain="triton_kernels",
        difficulty=LEVEL_DIFFICULTY.get(level, Difficulty.MEDIUM),
    )


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def run_curator(
    fs_backend: FSBackend,
    queue: FSQueue,
    *,
    n: int = 100,
    levels: Optional[list[str]] = None,
    seed: Optional[int] = None,
    max_pending: int = 100,
    conversation_path: Optional[Path] = None,
) -> list[Problem]:
    """Load KernelBench problems, dedup, save, and enqueue.

    Returns the list of newly created problems.
    """
    queue.initialize()

    def _log(event: str, **data: object) -> None:
        if conversation_path is not None:
            _append_jsonl(conversation_path, {
                "role": "system",
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **data,
            })

    _log("curator_start", n=n, levels=levels, max_pending=max_pending)

    # Stop early if the pending queue is already full enough
    pending_count = queue.count(MessageState.PENDING)
    if pending_count >= max_pending:
        logger.info(
            "Curator skipped: %d pending problems already in queue (max %d)",
            pending_count,
            max_pending,
        )
        _log("curator_skip", pending_count=pending_count)
        return []

    # Load existing problem titles for dedup
    existing_titles = {p.title for p in fs_backend.list_problems(limit=1000)}

    rows = load_kernelbench(levels=levels)
    _log("dataset_loaded", total_rows=len(rows))
    sampled = sample_problems(rows, n, seed=seed)

    created = []
    for row in sampled:
        # Check pending count before each enqueue
        if queue.count(MessageState.PENDING) >= max_pending:
            logger.info(
                "Curator stopping: pending queue reached %d (max %d)",
                max_pending,
                max_pending,
            )
            _log("queue_full", pending=max_pending)
            break

        problem = row_to_problem(row)
        if problem.title in existing_titles:
            logger.debug("Skipping duplicate: %s", problem.title)
            continue

        fs_backend.save_problem(problem)
        queue.enqueue(
            sender="curator",
            payload={
                "problem_id": problem.problem_id,
                "title": problem.title,
            },
        )
        existing_titles.add(problem.title)
        created.append(problem)
        _log("problem_created", problem_id=problem.problem_id, title=problem.title)

    logger.info(
        "Curator created %d problems (%d skipped as duplicates)",
        len(created),
        len(sampled) - len(created),
    )
    _log(
        "curator_done",
        created=len(created),
        skipped=len(sampled) - len(created),
    )
    return created
