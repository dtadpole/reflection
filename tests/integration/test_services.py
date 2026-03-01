"""Integration test: remote service deployment, health, and functionality.

Tests are organized in three layers:

1. **Service infrastructure** — SSH connectivity, systemd unit management, log access.
   These tests verify the deployment machinery works regardless of which service
   is deployed.

2. **kbEval service** (_one) — health endpoint, GPU device detection, eval round-trips
   with real PyTorch code.

3. **Text embedding service** (_two) — health endpoint, embedding round-trips,
   batch processing, dimension validation.

Requires:
- SSH access to _one (centos@1and1:41922) and _two (centos@1and1:42922)
- kbEval deployed and running on _one
- text-embedding deployed and running on _two
- SSH tunnels running: reflection services tunnel start

Run with:
    make test-services
"""

from __future__ import annotations

import textwrap

import pytest

from agenix.config import ServiceEndpoint, ServicesConfig, load_config
from services.deploy import ServiceDeployer
from services.health import HealthChecker
from services.kb_eval.baseline.client import KbEvalClient
from services.models import ServiceStatus
from services.text_embedding.baseline.client import TextEmbeddingClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    """Load real config with hosts.yaml endpoints."""
    return load_config()


@pytest.fixture(scope="module")
def endpoint_one(config) -> ServiceEndpoint:
    """Get the _one endpoint from config."""
    for ep in config.services.endpoints:
        if ep.name == "_one":
            return ep
    pytest.skip("Endpoint _one not configured in hosts.yaml")


@pytest.fixture(scope="module")
def services_config(config) -> ServicesConfig:
    return config.services


@pytest.fixture(scope="module")
def health_checker(services_config) -> HealthChecker:
    return HealthChecker(services_config)


@pytest.fixture(scope="module")
def deployer(services_config) -> ServiceDeployer:
    return ServiceDeployer(services_config)


@pytest.fixture(scope="module")
def kb_client(endpoint_one) -> KbEvalClient:
    """KbEvalClient using endpoint config (base_url points to localhost via tunnel)."""
    return KbEvalClient(endpoint_one.kb_eval)


@pytest.fixture(scope="module")
def endpoint_two(config) -> ServiceEndpoint:
    """Get the _two endpoint from config."""
    for ep in config.services.endpoints:
        if ep.name == "_two":
            return ep
    pytest.skip("Endpoint _two not configured in hosts.yaml")


@pytest.fixture(scope="module")
def te_client(endpoint_two) -> TextEmbeddingClient:
    """TextEmbeddingClient using endpoint config (base_url via tunnel)."""
    return TextEmbeddingClient(endpoint_two.text_embedding)


async def _embedding_reachable(te_client: TextEmbeddingClient) -> bool:
    """Check if text embedding is reachable (expects SSH tunnel running)."""
    try:
        health = await te_client.health()
        return health.status == ServiceStatus.RUNNING
    except Exception:
        return False


async def _service_reachable(kb_client: KbEvalClient) -> bool:
    """Check if kbEval is reachable (expects SSH tunnel running)."""
    try:
        health = await kb_client.health()
        return health.status == ServiceStatus.RUNNING
    except Exception:
        return False


# ===========================================================================
# 1. Service infrastructure (SSH + systemd)
# ===========================================================================


class TestSSHConnectivity:
    """Verify SSH access to remote hosts."""

    @pytest.mark.asyncio
    async def test_ssh_reachable(self, health_checker, endpoint_one):
        """SSH connection to _one should succeed."""
        ok = await health_checker.check_ssh(endpoint_one)
        assert ok, (
            f"SSH to {endpoint_one.name} ({endpoint_one.host}:{endpoint_one.port}) failed"
        )


class TestSystemdDeployment:
    """Verify systemd service management on remote hosts."""

    @pytest.mark.asyncio
    async def test_service_unit_exists(self, deployer, endpoint_one):
        """systemd should know about the kb-eval unit."""
        status_output = await deployer.systemd_status_kb_eval(endpoint_one)
        assert "kb-eval" in status_output, (
            f"kb-eval unit not found in systemd output: {status_output}"
        )

    @pytest.mark.asyncio
    async def test_service_is_active(self, deployer, endpoint_one):
        """kb-eval should be active (running)."""
        status_output = await deployer.systemd_status_kb_eval(endpoint_one)
        assert "active (running)" in status_output, (
            f"Service not active: {status_output}"
        )

    @pytest.mark.asyncio
    async def test_logs_available(self, deployer, endpoint_one):
        """Journal logs should be fetchable."""
        logs = await deployer.logs_kb_eval(endpoint_one, lines=10)
        assert len(logs) > 0, "No logs returned"
        assert "Failed to fetch" not in logs, f"Log fetch failed: {logs}"


# ===========================================================================
# 2. kbEval service
# ===========================================================================


class TestKbEvalHealth:
    """Test kbEval health endpoint via SSH tunnel."""

    @pytest.mark.asyncio
    async def test_health_running(self, kb_client):
        """Health endpoint should return running status."""
        if not await _service_reachable(kb_client):
            pytest.skip("kbEval not reachable (check SSH tunnel)")
        health = await kb_client.health()
        assert health.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_health_has_gpu(self, kb_client):
        """Should report at least one CUDA device."""
        if not await _service_reachable(kb_client):
            pytest.skip("kbEval not reachable")
        health = await kb_client.health()
        assert len(health.devices) > 0, "No devices reported"
        assert any("cuda" in d for d in health.devices), (
            f"No CUDA device in: {health.devices}"
        )

    @pytest.mark.asyncio
    async def test_health_no_pending(self, kb_client):
        """No pending requests when idle."""
        if not await _service_reachable(kb_client):
            pytest.skip("kbEval not reachable")
        health = await kb_client.health()
        assert health.pending_requests == 0


SIMPLE_REFERENCE = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    batch_size = 16
    dim = 16384

    def get_inputs():
        x = torch.rand(batch_size, dim)
        return [x]

    def get_init_inputs():
        return []
""")

SIMPLE_CORRECT = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class ModelNew(nn.Module):
        def __init__(self):
            super(ModelNew, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)
""")

SIMPLE_WRONG = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class ModelNew(nn.Module):
        def __init__(self):
            super(ModelNew, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)
""")


class TestKbEvalPyTorch:
    """Test kbEval eval round-trips with PyTorch code."""

    @pytest.mark.asyncio
    async def test_eval_correct(self, kb_client):
        """Correct generated code should pass compilation and correctness."""
        if not await _service_reachable(kb_client):
            pytest.skip("kbEval not reachable")
        result = await kb_client.eval(
            reference_code=SIMPLE_REFERENCE,
            generated_code=SIMPLE_CORRECT,
            code_type="pytorch",
        )
        assert result.compiled is True, f"Should compile: {result}"
        assert result.correctness is True, f"Should be correct: {result}"
        assert result.runtime > 0, f"Should have runtime: {result}"

    @pytest.mark.asyncio
    async def test_eval_wrong(self, kb_client):
        """Wrong generated code should compile but fail correctness."""
        if not await _service_reachable(kb_client):
            pytest.skip("kbEval not reachable")
        result = await kb_client.eval(
            reference_code=SIMPLE_REFERENCE,
            generated_code=SIMPLE_WRONG,
            code_type="pytorch",
        )
        assert result.compiled is True, f"Should compile: {result}"
        assert result.correctness is False, f"Should be incorrect: {result}"

    @pytest.mark.asyncio
    async def test_eval_ref_only(self, kb_client):
        """eval_ref should benchmark reference code only."""
        if not await _service_reachable(kb_client):
            pytest.skip("kbEval not reachable")
        result = await kb_client.eval_ref(
            reference_code=SIMPLE_REFERENCE,
        )
        assert result.compiled is True, f"Reference should compile: {result}"
        assert result.runtime > 0, f"Should have runtime: {result}"


# ===========================================================================
# 3. Text embedding service (_two)
# ===========================================================================


class TestTextEmbeddingSystemd:
    """Verify text-embedding systemd deployment on _two."""

    @pytest.mark.asyncio
    async def test_ssh_reachable(self, health_checker, endpoint_two):
        """SSH connection to _two should succeed."""
        ok = await health_checker.check_ssh(endpoint_two)
        assert ok, (
            f"SSH to {endpoint_two.name} ({endpoint_two.host}:{endpoint_two.port}) failed"
        )

    @pytest.mark.asyncio
    async def test_service_unit_exists(self, deployer, endpoint_two):
        """systemd should know about the text-embedding unit."""
        status_output = await deployer.systemd_status_text_embedding(endpoint_two)
        assert "text-embedding" in status_output, (
            f"text-embedding unit not found: {status_output}"
        )

    @pytest.mark.asyncio
    async def test_service_is_active(self, deployer, endpoint_two):
        """text-embedding should be active (running)."""
        status_output = await deployer.systemd_status_text_embedding(endpoint_two)
        assert "active (running)" in status_output, (
            f"Service not active: {status_output}"
        )

    @pytest.mark.asyncio
    async def test_logs_available(self, deployer, endpoint_two):
        """Journal logs should be fetchable."""
        logs = await deployer.logs_text_embedding(endpoint_two, lines=10)
        assert len(logs) > 0, "No logs returned"
        assert "Failed to fetch" not in logs, f"Log fetch failed: {logs}"


class TestTextEmbeddingHealth:
    """Test text embedding health endpoint via SSH tunnel."""

    @pytest.mark.asyncio
    async def test_health_running(self, te_client):
        """Health endpoint should return running status."""
        if not await _embedding_reachable(te_client):
            pytest.skip("text-embedding not reachable (check SSH tunnel)")
        health = await te_client.health()
        assert health.status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_health_has_gpu(self, te_client):
        """Should report at least one CUDA device."""
        if not await _embedding_reachable(te_client):
            pytest.skip("text-embedding not reachable")
        health = await te_client.health()
        assert len(health.devices) > 0, "No devices reported"
        assert any("cuda" in d for d in health.devices), (
            f"No CUDA device in: {health.devices}"
        )


class TestTextEmbeddingEmbed:
    """Test text embedding round-trips."""

    @pytest.mark.asyncio
    async def test_embed_single(self, te_client):
        """Single text should return one embedding vector."""
        if not await _embedding_reachable(te_client):
            pytest.skip("text-embedding not reachable")
        result = await te_client.embed(["Hello world"])
        assert len(result.embeddings) == 1
        assert result.dimension == 4096
        assert len(result.embeddings[0]) == 4096
        assert result.model == "Qwen/Qwen3-Embedding-8B"

    @pytest.mark.asyncio
    async def test_embed_batch(self, te_client):
        """Multiple texts should return one vector per text."""
        if not await _embedding_reachable(te_client):
            pytest.skip("text-embedding not reachable")
        texts = [
            "GPU kernel optimization",
            "memory coalescing patterns",
            "Triton language tutorial",
        ]
        result = await te_client.embed(texts)
        assert len(result.embeddings) == 3
        for vec in result.embeddings:
            assert len(vec) == 4096

    @pytest.mark.asyncio
    async def test_embed_distinct_vectors(self, te_client):
        """Different texts should produce different embeddings."""
        if not await _embedding_reachable(te_client):
            pytest.skip("text-embedding not reachable")
        result = await te_client.embed(["cat", "quantum computing"])
        v0, v1 = result.embeddings
        # Cosine similarity should be well below 1.0
        import math

        dot = sum(a * b for a, b in zip(v0, v1))
        mag0 = math.sqrt(sum(x * x for x in v0))
        mag1 = math.sqrt(sum(x * x for x in v1))
        cosine = dot / (mag0 * mag1)
        assert cosine < 0.95, f"Vectors too similar: cosine={cosine:.4f}"

    @pytest.mark.asyncio
    async def test_embed_with_instruction(self, te_client):
        """Embedding with instruction prefix should succeed."""
        if not await _embedding_reachable(te_client):
            pytest.skip("text-embedding not reachable")
        result = await te_client.embed(
            ["Triton kernel for matrix multiply"],
            instruction="Represent the code concept for retrieval:",
        )
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 4096
