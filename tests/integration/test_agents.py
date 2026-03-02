"""Integration tests for async agent handlers.

Tests each agent type against live infrastructure:
- CURATOR: Loads real KernelBench data from HuggingFace
- SOLVER: Runs real Claude agent with MCP tools (verifier + retriever)
- CRITIC: Runs real Claude agent on an experience
- ORGANIZER: Runs real Claude agent on experiences + reflection cards
- INSIGHT_FINDER: Runs real Claude agent on experience batches

Requires:
- HuggingFace dataset access (for curator)
- SSH tunnels running: reflection services tunnel start
- kbEval service on _one (for solver verifier)
- text_embedding + reranker on _two (for solver retriever)
- Claude API key (for solver, critic, organizer, insight_finder)

Run with:
    uv run pytest tests/integration/test_agents.py -v -s
    uv run pytest tests/integration/test_agents.py -v -s -k curator  # Just curator
"""

from __future__ import annotations

import os

import pytest

from agenix.config import load_config
from agenix.queue.fs_queue import FSQueue
from agenix.queue.models import MessageState
from agenix.storage.fs_backend import FSBackend
from agenix.storage.models import Difficulty, Problem, ProblemStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    """Load config with test environment."""
    cfg = load_config()
    cfg.storage.env = f"test_{os.environ.get('USER', 'ci')}"
    return cfg


@pytest.fixture(scope="module")
def fs_backend(config):
    fs = FSBackend(config.storage)
    fs.initialize()
    return fs


@pytest.fixture(scope="module")
def problems_queue(config):
    q = FSQueue("problems", config.storage)
    q.initialize()
    return q


@pytest.fixture(scope="module")
def experiences_queue(config):
    q = FSQueue("experiences", config.storage)
    q.initialize()
    return q


# ---------------------------------------------------------------------------
# Curator Integration Tests
# ---------------------------------------------------------------------------


class TestCuratorIntegration:
    """Tests curator against real KernelBench dataset from HuggingFace."""

    def test_load_kernelbench_level_1(self):
        """Load level_1 problems from HuggingFace."""
        from agenix.agents.curator_handler import load_kernelbench

        rows = load_kernelbench(levels=["level_1"])
        assert len(rows) > 0
        # level_1 has 100 problems
        assert len(rows) >= 50  # at least 50, allowing for dataset updates

        # Verify row structure
        row = rows[0]
        assert "code" in row
        assert "level" in row
        assert "name" in row
        assert row["level"] == "level_1"
        assert "class Model" in row["code"]

    def test_load_and_enqueue(self, fs_backend, problems_queue):
        """Load 5 problems and enqueue them."""
        from agenix.agents.curator_handler import run_curator

        problems = run_curator(
            fs_backend, problems_queue, n=5, levels=["level_1"], seed=42
        )
        assert len(problems) > 0
        assert len(problems) <= 5

        # Verify all have correct domain
        for p in problems:
            assert p.domain == "triton_kernels"
            assert p.reference_code != ""
            assert "class Model" in p.reference_code

        # Verify queue has messages
        assert problems_queue.count(MessageState.PENDING) >= len(problems)

    def test_dedup_on_rerun(self, fs_backend, problems_queue):
        """Running again with same seed should dedup."""
        from agenix.agents.curator_handler import run_curator

        before = problems_queue.count(MessageState.PENDING)
        problems = run_curator(
            fs_backend, problems_queue, n=5, levels=["level_1"], seed=42
        )
        # All should be skipped as duplicates
        assert len(problems) == 0
        assert problems_queue.count(MessageState.PENDING) == before


# ---------------------------------------------------------------------------
# Solver Integration Tests
# ---------------------------------------------------------------------------


class TestSolverIntegration:
    """Tests solver against real Claude API + verifier + retriever."""

    @pytest.fixture(scope="class")
    def solver_deps(self, config, fs_backend, problems_queue, experiences_queue):
        """Set up solver dependencies with live services."""
        from agenix.runner import ClaudeRunner
        from agenix.tools.loader import load_tool
        from agenix.tools.registry import ToolRegistry
        from tools.knowledge.baseline.embedder import RemoteEmbedder
        from tools.knowledge.baseline.index import LanceIndex
        from tools.knowledge.baseline.store import KnowledgeStore

        ep_one = ep_two = None
        for ep in config.services.endpoints:
            if ep.name == "_one":
                ep_one = ep
            if ep.name == "_two":
                ep_two = ep

        if not ep_two:
            pytest.skip("Endpoint _two not configured")
        if not ep_one:
            pytest.skip("Endpoint _one not configured")

        embedder = RemoteEmbedder(config=ep_two.text_embedding, dimension=4096)
        lance = LanceIndex(db_path=config.storage.lance_path, vector_dim=4096)
        store = KnowledgeStore(
            config=config, fs_backend=fs_backend,
            lance_index=lance, embedder=embedder,
        )
        store.initialize()

        registry = ToolRegistry()

        from services.reranker.baseline.client import RerankerClient

        rr_client = RerankerClient(ep_two.reranker)
        retriever_def = load_tool("retriever", variant="rerank")
        registry.register(
            retriever_def.create_fn(knowledge_store=store, reranker_client=rr_client)
        )

        from services.kb_eval.baseline.client import KbEvalClient

        kb_client = KbEvalClient(ep_one.kb_eval)
        verifier_def = load_tool("verifier", variant="kb_eval")
        registry.register(verifier_def.create_fn(kb_eval_client=kb_client))

        runner = ClaudeRunner(tool_registry=registry)

        return runner, fs_backend, store, experiences_queue

    def test_solve_simple_kernel(self, solver_deps, problems_queue, config):
        """Solver should attempt to solve a level_1 problem."""
        runner, fs_backend, store, exp_queue = solver_deps

        from agenix.agents.solver_handler import SolverHandler

        # Create a simple test problem
        problem = Problem(
            title="[KernelBench/level_1] Test ReLU",
            description=(
                "Convert the following PyTorch code to Triton:\n"
                "```python\n"
                "class Model(nn.Module):\n"
                "    def forward(self, x): return torch.relu(x)\n"
                "```"
            ),
            reference_code=(
                "import torch\nimport torch.nn as nn\n\n"
                "class Model(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "    def forward(self, x):\n"
                "        return torch.relu(x)\n\n"
                "batch_size = 16\ndim = 16384\n\n"
                "def get_inputs():\n"
                "    return [torch.rand(batch_size, dim)]\n\n"
                "def get_init_inputs():\n"
                "    return []\n"
            ),
            domain="triton_kernels",
            difficulty=Difficulty.EASY,
        )
        fs_backend.save_problem(problem)

        # Enqueue the problem
        problems_queue.enqueue(
            "test",
            {"problem_id": problem.problem_id, "title": problem.title},
        )

        # Dequeue and process
        message = problems_queue.dequeue()
        assert message is not None

        handler = SolverHandler(
            runner=runner,
            fs_backend=fs_backend,
            knowledge_store=store,
            experiences_queue=exp_queue,
        )

        try:
            handler.handle(message)
            problems_queue.complete(message.message_id)

            # Verify experience was enqueued
            assert exp_queue.count(MessageState.PENDING) > 0

            # Verify problem status was updated
            updated = fs_backend.get_problem(problem.problem_id)
            assert updated.status in (ProblemStatus.SOLVED, ProblemStatus.FAILED)
        except Exception as e:
            problems_queue.fail(message.message_id, str(e))
            pytest.fail(f"Solver handler failed: {e}")


# ---------------------------------------------------------------------------
# Critic Integration Tests
# ---------------------------------------------------------------------------


class TestCriticIntegration:
    """Tests critic against real Claude API."""

    def test_critique_experience(self, config, fs_backend, experiences_queue):
        """Critic should produce reflection cards from an experience."""
        from agenix.agents.critic_handler import CriticHandler
        from agenix.runner import ClaudeRunner
        from agenix.storage.models import Experience
        from tools.knowledge.baseline.store import KnowledgeStore

        store = KnowledgeStore(config=config, fs_backend=fs_backend)
        store.initialize()

        runner = ClaudeRunner()

        # Create test problem and experience
        problem = Problem(
            title="Test Problem for Critic",
            description="A simple test problem",
            domain="triton_kernels",
            difficulty=Difficulty.EASY,
        )
        fs_backend.save_problem(problem)

        experience = Experience(
            problem_id=problem.problem_id,
            code_solution="class ModelNew(nn.Module): pass",
            final_answer="Simple passthrough",
            is_correct=False,
        )
        fs_backend.save_experience(experience)

        # Enqueue experience message
        experiences_queue.enqueue(
            "test",
            {
                "experience_id": experience.experience_id,
                "problem_id": problem.problem_id,
            },
        )

        message = experiences_queue.dequeue()
        assert message is not None

        handler = CriticHandler(
            runner=runner,
            fs_backend=fs_backend,
            knowledge_store=store,
        )

        try:
            handler.handle(message)
            experiences_queue.complete(message.message_id)
            # If we get here, the critic produced output successfully
        except Exception as e:
            experiences_queue.fail(message.message_id, str(e))
            pytest.fail(f"Critic handler failed: {e}")


# ---------------------------------------------------------------------------
# Organizer Integration Tests
# ---------------------------------------------------------------------------


class TestOrganizerIntegration:
    """Tests organizer against real Claude API."""

    def test_organize_knowledge(self, config, fs_backend):
        """Organizer should produce knowledge cards from experiences."""
        from agenix.agents.organizer_handler import OrganizerHandler
        from agenix.runner import ClaudeRunner
        from agenix.storage.models import Experience
        from tools.knowledge.baseline.store import KnowledgeStore

        store = KnowledgeStore(config=config, fs_backend=fs_backend)
        store.initialize()

        runner = ClaudeRunner()

        # Create a problem + experience for the organizer to analyze
        problem = Problem(
            title="Test Problem for Organizer",
            description="GPU kernel optimization problem",
            domain="triton_kernels",
            difficulty=Difficulty.MEDIUM,
        )
        fs_backend.save_problem(problem)

        experience = Experience(
            problem_id=problem.problem_id,
            code_solution="@triton.jit\ndef kernel(): pass\n\nclass ModelNew: pass",
            final_answer="Used tiling strategy for matmul",
            is_correct=True,
        )
        fs_backend.save_experience(experience)

        handler = OrganizerHandler(
            runner=runner,
            fs_backend=fs_backend,
            knowledge_store=store,
        )

        # Should not raise
        handler.handle()


# ---------------------------------------------------------------------------
# Insight Finder Integration Tests
# ---------------------------------------------------------------------------


class TestInsightFinderIntegration:
    """Tests insight finder against real Claude API."""

    def test_find_insights(self, config, fs_backend):
        """Insight finder should produce insight cards from experiences."""
        from agenix.agents.insight_handler import InsightHandler
        from agenix.runner import ClaudeRunner
        from agenix.storage.models import Experience
        from tools.knowledge.baseline.store import KnowledgeStore

        store = KnowledgeStore(config=config, fs_backend=fs_backend)
        store.initialize()

        runner = ClaudeRunner()

        # Create multiple experiences for pattern detection
        for i in range(3):
            problem = Problem(
                title=f"Insight Test Problem {i}",
                description=f"GPU kernel problem variant {i}",
                domain="triton_kernels",
                difficulty=Difficulty.MEDIUM,
            )
            fs_backend.save_problem(problem)

            experience = Experience(
                problem_id=problem.problem_id,
                code_solution=f"# Kernel {i}\nclass ModelNew: pass",
                final_answer=f"Attempt {i}",
                is_correct=(i % 2 == 0),
            )
            fs_backend.save_experience(experience)

        handler = InsightHandler(
            runner=runner,
            fs_backend=fs_backend,
            knowledge_store=store,
        )

        # Should not raise
        handler.handle()
