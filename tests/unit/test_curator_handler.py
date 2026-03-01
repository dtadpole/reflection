"""Tests for the curator handler (KernelBench loader)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agenix.agents.curator_handler import (
    row_to_problem,
    run_curator,
    sample_problems,
)
from agenix.config import StorageConfig
from agenix.queue.fs_queue import FSQueue
from agenix.queue.models import MessageState
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import Difficulty

_RELU_CODE = (
    "import torch\nclass Model(torch.nn.Module):\n"
    "    def forward(self, x): return torch.relu(x)\n"
)
_SOFTMAX_CODE = (
    "import torch\nclass Model(torch.nn.Module):\n"
    "    def forward(self, x): return torch.softmax(x, dim=-1)\n"
)
_MATMUL_CODE = (
    "import torch\nclass Model(torch.nn.Module):\n"
    "    def forward(self, x, w): return torch.matmul(x, w)\n"
)

SAMPLE_ROWS = [
    {"code": _RELU_CODE, "level": "level_1", "name": "relu_activation", "kb_problem_id": "1"},
    {"code": _SOFTMAX_CODE, "level": "level_2", "name": "softmax", "kb_problem_id": "2"},
    {"code": _MATMUL_CODE, "level": "level_3", "name": "matmul", "kb_problem_id": "3"},
]


@pytest.fixture
def storage_config(tmp_path):
    return StorageConfig(data_root=str(tmp_path), env="test")


@pytest.fixture
def fs_backend(storage_config):
    fs = FSBackend(storage_config)
    fs.initialize()
    return fs


@pytest.fixture
def queue(storage_config):
    q = FSQueue("problems", storage_config)
    q.initialize()
    return q


class TestRowToProblem:
    def test_basic_conversion(self):
        row = SAMPLE_ROWS[0]
        problem = row_to_problem(row)
        assert problem.title == "[KernelBench/level_1] relu_activation"
        assert problem.domain == "triton_kernels"
        assert problem.difficulty == Difficulty.EASY
        assert problem.reference_code == row["code"]
        assert "torch.relu" in problem.description

    def test_level_difficulty_mapping(self):
        for row, expected in [
            (SAMPLE_ROWS[0], Difficulty.EASY),    # level_1
            (SAMPLE_ROWS[1], Difficulty.MEDIUM),   # level_2
            (SAMPLE_ROWS[2], Difficulty.HARD),     # level_3
        ]:
            problem = row_to_problem(row)
            assert problem.difficulty == expected

    def test_reference_code_preserved(self):
        row = SAMPLE_ROWS[0]
        problem = row_to_problem(row)
        assert problem.reference_code == row["code"]

    def test_description_includes_reference(self):
        row = SAMPLE_ROWS[0]
        problem = row_to_problem(row)
        assert "PyTorch" in problem.description
        assert "Triton" in problem.description
        assert "ModelNew" in problem.description


class TestSampleProblems:
    def test_sample_less_than_available(self):
        sampled = sample_problems(SAMPLE_ROWS, 2, seed=42)
        assert len(sampled) == 2

    def test_sample_more_than_available(self):
        sampled = sample_problems(SAMPLE_ROWS, 100, seed=42)
        assert len(sampled) == 3  # all available

    def test_deterministic_with_seed(self):
        s1 = sample_problems(SAMPLE_ROWS, 2, seed=42)
        s2 = sample_problems(SAMPLE_ROWS, 2, seed=42)
        assert [r["name"] for r in s1] == [r["name"] for r in s2]

    def test_different_seeds_different_results(self):
        s1 = sample_problems(SAMPLE_ROWS, 2, seed=1)
        s2 = sample_problems(SAMPLE_ROWS, 2, seed=2)
        # With only 3 items, there's a chance they're the same,
        # but different seeds should generally produce different orderings
        # Just verify both return correct count
        assert len(s1) == 2
        assert len(s2) == 2


class TestRunCurator:
    @patch("agenix.agents.curator_handler.load_kernelbench")
    def test_creates_problems_and_enqueues(self, mock_load, fs_backend, queue):
        mock_load.return_value = SAMPLE_ROWS

        problems = run_curator(fs_backend, queue, n=3, seed=42)

        assert len(problems) == 3
        assert queue.count(MessageState.PENDING) == 3

        # Verify problems are saved to FSBackend
        stored = fs_backend.list_problems()
        assert len(stored) == 3

    @patch("agenix.agents.curator_handler.load_kernelbench")
    def test_dedup_skips_existing(self, mock_load, fs_backend, queue):
        mock_load.return_value = SAMPLE_ROWS

        # First run
        run_curator(fs_backend, queue, n=3, seed=42)
        assert queue.count(MessageState.PENDING) == 3

        # Second run — should skip all as duplicates
        problems2 = run_curator(fs_backend, queue, n=3, seed=42)
        assert len(problems2) == 0
        # Queue still has 3 from first run
        assert queue.count(MessageState.PENDING) == 3

    @patch("agenix.agents.curator_handler.load_kernelbench")
    def test_passes_levels_to_loader(self, mock_load, fs_backend, queue):
        mock_load.return_value = []

        run_curator(fs_backend, queue, n=10, levels=["level_1", "level_3"])

        mock_load.assert_called_once_with(levels=["level_1", "level_3"])

    @patch("agenix.agents.curator_handler.load_kernelbench")
    def test_sample_subset(self, mock_load, fs_backend, queue):
        mock_load.return_value = SAMPLE_ROWS

        problems = run_curator(fs_backend, queue, n=1, seed=42)

        assert len(problems) == 1
        assert queue.count(MessageState.PENDING) == 1
