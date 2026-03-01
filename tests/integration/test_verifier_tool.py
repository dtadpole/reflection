"""Integration test: verifier tool (kb_eval variant) against live kbEval service.

Requires:
- kbEval running on a remote host
- SSH tunnels running: reflection services tunnel start
- GPU available on the remote host

Run with:
    uv run pytest tests/integration/test_verifier_tool.py -v -s
"""

from __future__ import annotations

import json
import textwrap

import pytest

from agenix.config import load_config
from agenix.tools.loader import load_tool
from services.kb_eval.baseline.client import KbEvalClient


@pytest.fixture(scope="module")
def kb_client() -> KbEvalClient:
    """Create a KbEvalClient using endpoint config (base_url via tunnel)."""
    cfg = load_config()
    for ep in cfg.services.endpoints:
        if ep.name == "_one":
            return KbEvalClient(ep.kb_eval)
    pytest.skip("Endpoint _one not configured in hosts.yaml")


@pytest.fixture(scope="module")
def verifier_tool(kb_client):
    """Load the verifier tool via the tool loader and create it with a real client."""
    tool_def = load_tool("verifier", variant="kb_eval")
    return tool_def.create_fn(kb_eval_client=kb_client)


async def _check_service(kb_client: KbEvalClient) -> bool:
    """Check if kbEval is reachable via SSH tunnel."""
    try:
        health = await kb_client.health()
        return health.status.value == "running"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PyTorch: identity test (generated == reference → must be correct)
# ---------------------------------------------------------------------------

PYTORCH_RELU_REFERENCE = textwrap.dedent("""\
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

PYTORCH_RELU_IDENTITY = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class ModelNew(nn.Module):
        def __init__(self):
            super(ModelNew, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)
""")


class TestVerifierPyTorch:
    """Test verifier with pure PyTorch code (no Triton)."""

    @pytest.mark.asyncio
    async def test_service_healthy(self, kb_client):
        """Verify kbEval is reachable before running other tests."""
        ok = await _check_service(kb_client)
        if not ok:
            pytest.skip("kbEval service not reachable (check SSH tunnel)")

    @pytest.mark.asyncio
    async def test_pytorch_identity_correct(self, verifier_tool, kb_client):
        """Generated code == reference code → must compile and be correct."""
        if not await _check_service(kb_client):
            pytest.skip("kbEval not reachable")

        result = await verifier_tool.handler({
            "reference_code": PYTORCH_RELU_REFERENCE,
            "generated_code": PYTORCH_RELU_IDENTITY,
            "code_type": "pytorch",
        })

        data = _parse_result(result)
        assert data["compiled"] is True, f"Expected compiled=True, got: {data}"
        assert data["correctness"] is True, f"Expected correctness=True, got: {data}"
        assert data["runtime"] > 0, f"Expected runtime > 0, got: {data['runtime']}"

    @pytest.mark.asyncio
    async def test_pytorch_wrong_output(self, verifier_tool, kb_client):
        """Generated code returns zeros → should compile but fail correctness."""
        if not await _check_service(kb_client):
            pytest.skip("kbEval not reachable")

        wrong_code = textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class ModelNew(nn.Module):
                def __init__(self):
                    super(ModelNew, self).__init__()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.zeros_like(x)
        """)

        result = await verifier_tool.handler({
            "reference_code": PYTORCH_RELU_REFERENCE,
            "generated_code": wrong_code,
            "code_type": "pytorch",
        })

        data = _parse_result(result)
        assert data["compiled"] is True, f"Should compile, got: {data}"
        assert data["correctness"] is False, f"Should be incorrect, got: {data}"

    @pytest.mark.asyncio
    async def test_pytorch_syntax_error(self, verifier_tool, kb_client):
        """Syntax error in generated code → should not compile."""
        if not await _check_service(kb_client):
            pytest.skip("kbEval not reachable")

        bad_code = textwrap.dedent("""\
            import torch

            class ModelNew(torch.nn.Module:  # syntax error
                def forward(self, x):
                    return x
        """)

        result = await verifier_tool.handler({
            "reference_code": PYTORCH_RELU_REFERENCE,
            "generated_code": bad_code,
            "code_type": "pytorch",
        })

        data = _parse_result(result)
        assert data["compiled"] is False, f"Should not compile, got: {data}"


# ---------------------------------------------------------------------------
# Triton: real kernel test
# ---------------------------------------------------------------------------

TRITON_RELU_GENERATED = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        output = tl.maximum(x, 0.0)
        tl.store(out_ptr + offsets, output, mask=mask)

    class ModelNew(torch.nn.Module):
        def __init__(self):
            super(ModelNew, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n_elements = x.numel()
            BLOCK_SIZE = 1024
            grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return out
""")


class TestVerifierTriton:
    """Test verifier with Triton kernel code."""

    @pytest.mark.asyncio
    async def test_triton_relu_correct(self, verifier_tool, kb_client):
        """Triton ReLU kernel should compile and produce correct results."""
        if not await _check_service(kb_client):
            pytest.skip("kbEval not reachable")

        result = await verifier_tool.handler({
            "reference_code": PYTORCH_RELU_REFERENCE,
            "generated_code": TRITON_RELU_GENERATED,
            "code_type": "triton",
        })

        data = _parse_result(result)
        assert data["compiled"] is True, f"Triton kernel should compile, got: {data}"
        assert data["correctness"] is True, f"Triton ReLU should be correct, got: {data}"
        assert data["runtime"] > 0, f"Expected runtime > 0, got: {data['runtime']}"

    @pytest.mark.asyncio
    async def test_triton_wrong_kernel(self, verifier_tool, kb_client):
        """Triton kernel that negates input → should compile but fail correctness."""
        if not await _check_service(kb_client):
            pytest.skip("kbEval not reachable")

        wrong_triton = textwrap.dedent("""\
            import torch
            import triton
            import triton.language as tl

            @triton.jit
            def negate_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                output = -x  # negate instead of relu
                tl.store(out_ptr + offsets, output, mask=mask)

            class ModelNew(torch.nn.Module):
                def __init__(self):
                    super(ModelNew, self).__init__()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    out = torch.empty_like(x)
                    n_elements = x.numel()
                    BLOCK_SIZE = 1024
                    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
                    negate_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
                    return out
        """)

        result = await verifier_tool.handler({
            "reference_code": PYTORCH_RELU_REFERENCE,
            "generated_code": wrong_triton,
            "code_type": "triton",
        })

        data = _parse_result(result)
        assert data["compiled"] is True, f"Should compile, got: {data}"
        assert data["correctness"] is False, f"Should be incorrect, got: {data}"


# ---------------------------------------------------------------------------
# Tool loader integration
# ---------------------------------------------------------------------------


class TestToolLoaderIntegration:
    """Test that load_tool() correctly loads the verifier variant."""

    def test_load_verifier_kb_eval(self):
        """load_tool('verifier', variant='kb_eval') should load correctly."""
        tool_def = load_tool("verifier", variant="kb_eval")
        assert tool_def.name == "verifier"
        assert tool_def.variant == "kb_eval"
        assert "GPU kernel" in tool_def.description
        assert callable(tool_def.create_fn)

    def test_load_retriever_baseline(self):
        """load_tool('retriever', variant='baseline') should load correctly."""
        tool_def = load_tool("retriever", variant="baseline")
        assert tool_def.name == "retriever"
        assert tool_def.variant == "baseline"
        assert "knowledge base" in tool_def.description.lower()
        assert callable(tool_def.create_fn)


# ---------------------------------------------------------------------------
# Input validation via the tool (no service needed)
# ---------------------------------------------------------------------------


class TestVerifierValidation:
    """Test input validation in the verifier tool (no remote call needed)."""

    @pytest.mark.asyncio
    async def test_empty_reference_code(self, verifier_tool):
        """Empty reference code should return an error."""
        result = await verifier_tool.handler({
            "reference_code": "",
            "generated_code": "class ModelNew: pass",
        })
        assert result.get("is_error") is True

    @pytest.mark.asyncio
    async def test_empty_generated_code(self, verifier_tool):
        """Empty generated code should return an error."""
        result = await verifier_tool.handler({
            "reference_code": "class Model: pass",
            "generated_code": "",
        })
        assert result.get("is_error") is True

    @pytest.mark.asyncio
    async def test_invalid_code_type(self, verifier_tool):
        """Invalid code_type should return an error."""
        result = await verifier_tool.handler({
            "reference_code": "class Model: pass",
            "generated_code": "class ModelNew: pass",
            "code_type": "java",
        })
        assert result.get("is_error") is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result: dict) -> dict:
    """Parse MCP tool result JSON."""
    if result.get("is_error"):
        text = result["content"][0]["text"]
        raise AssertionError(f"Tool returned error: {text}")
    text = result["content"][0]["text"]
    return json.loads(text)
