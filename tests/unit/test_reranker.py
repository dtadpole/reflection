"""Tests for the reranker service: models, scoring logic, client, and server."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenix.config import RerankerClientConfig, RerankerServerConfig
from services.models import RerankResult, ServiceStatus
from services.reranker.baseline.client import RerankerClient
from services.reranker.baseline.server import (
    _build_prompt,
    _extract_score,
)

# --- RerankResult model ---


class TestRerankResult:
    def test_basic(self):
        r = RerankResult(scores=[0.9, 0.1])
        assert r.scores == [0.9, 0.1]
        assert r.model == ""

    def test_with_model(self):
        r = RerankResult(scores=[0.5], model="Qwen/Qwen3-32B")
        assert r.model == "Qwen/Qwen3-32B"

    def test_empty_scores(self):
        r = RerankResult(scores=[])
        assert r.scores == []

    def test_roundtrip(self):
        r = RerankResult(scores=[0.95, 0.02, 0.5], model="test-model")
        data = r.model_dump()
        r2 = RerankResult.model_validate(data)
        assert r2.scores == r.scores
        assert r2.model == r.model


# --- Score extraction logic ---


class TestExtractScore:
    def test_both_tokens_present(self):
        """When both yes and no logprobs are present, compute normalized score."""
        # P(yes) = exp(-0.1) ≈ 0.905, P(no) = exp(-3.0) ≈ 0.050
        logprobs = {"yes": -0.1, "no": -3.0}
        score = _extract_score(logprobs)
        expected = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-3.0))
        assert abs(score - expected) < 1e-6
        assert score > 0.9  # strongly yes

    def test_strongly_no(self):
        """When no is much more likely than yes."""
        logprobs = {"yes": -5.0, "no": -0.05}
        score = _extract_score(logprobs)
        assert score < 0.01  # strongly no

    def test_equal_probability(self):
        """When yes and no have equal logprobs, score should be 0.5."""
        logprobs = {"yes": -1.0, "no": -1.0}
        score = _extract_score(logprobs)
        assert abs(score - 0.5) < 1e-6

    def test_only_yes_present(self):
        """When only yes token is in logprobs."""
        logprobs = {"yes": -0.5, "maybe": -2.0}
        score = _extract_score(logprobs)
        assert score == 1.0

    def test_only_no_present(self):
        """When only no token is in logprobs."""
        logprobs = {"no": -0.5, "maybe": -2.0}
        score = _extract_score(logprobs)
        assert score == 0.0

    def test_neither_present(self):
        """When neither yes nor no is in logprobs, return 0.5."""
        logprobs = {"maybe": -0.5, "perhaps": -1.0}
        score = _extract_score(logprobs)
        assert score == 0.5

    def test_empty_logprobs(self):
        """Empty dict returns 0.5."""
        score = _extract_score({})
        assert score == 0.5

    def test_case_insensitive(self):
        """Tokens with whitespace/case variations are handled."""
        logprobs = {"Yes": -0.2, "No": -3.0}
        score = _extract_score(logprobs)
        expected = math.exp(-0.2) / (math.exp(-0.2) + math.exp(-3.0))
        assert abs(score - expected) < 1e-6

    def test_whitespace_in_token(self):
        """Tokens with leading/trailing whitespace are stripped."""
        logprobs = {" yes ": -0.3, " no ": -2.5}
        score = _extract_score(logprobs)
        expected = math.exp(-0.3) / (math.exp(-0.3) + math.exp(-2.5))
        assert abs(score - expected) < 1e-6

    def test_score_range(self):
        """Score is always between 0 and 1."""
        test_cases = [
            {"yes": -0.001, "no": -10.0},  # very high
            {"yes": -10.0, "no": -0.001},  # very low
            {"yes": -1.0, "no": -1.0},  # middle
            {"yes": -0.5},  # only yes
            {"no": -0.5},  # only no
        ]
        for logprobs in test_cases:
            score = _extract_score(logprobs)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {logprobs}"


# --- Prompt building ---


class TestBuildPrompt:
    def test_contains_query_and_document(self):
        prompt = _build_prompt("Find relevant docs", "matrix multiply", "Use tiling")
        assert "<Query>: matrix multiply" in prompt
        assert "<Document>: Use tiling" in prompt
        assert "<Instruct>: Find relevant docs" in prompt

    def test_chatml_format(self):
        prompt = _build_prompt("inst", "q", "d")
        assert prompt.startswith("<|im_start|>system")
        assert "<|im_end|>" in prompt
        assert "<|im_start|>assistant\n" in prompt

    def test_thinking_bypass(self):
        """Prompt includes think tags to bypass Qwen3 thinking mode."""
        prompt = _build_prompt("inst", "q", "d")
        assert "<think>\n\n</think>\n" in prompt
        assert prompt.endswith("<think>\n\n</think>\n")

    def test_yes_no_instruction_in_system(self):
        prompt = _build_prompt("inst", "q", "d")
        assert '"yes"' in prompt
        assert '"no"' in prompt


# --- Config classes ---


class TestRerankerServerConfig:
    def test_defaults(self):
        cfg = RerankerServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 42983
        assert cfg.vllm_port == 42984
        assert cfg.model_name == "Qwen/Qwen3-32B"
        assert cfg.device == "cuda:0"


class TestRerankerClientConfig:
    def test_defaults(self):
        cfg = RerankerClientConfig()
        assert cfg.base_url == "http://localhost:42983"
        assert cfg.timeout == 120
        assert cfg.retry_count == 3
        assert cfg.retry_interval == 2.0


# --- RerankerClient ---


class TestRerankerClientConstruction:
    def test_default_config(self):
        client = RerankerClient()
        assert client._base_url == "http://localhost:42983"

    def test_custom_config(self):
        cfg = RerankerClientConfig(base_url="http://gpu2:9999", timeout=30)
        client = RerankerClient(cfg)
        assert client._base_url == "http://gpu2:9999"
        assert client._config.timeout == 30

    def test_trailing_slash_stripped(self):
        cfg = RerankerClientConfig(base_url="http://gpu2:42983/")
        client = RerankerClient(cfg)
        assert client._base_url == "http://gpu2:42983"


class TestRerankerClientHealth:
    @pytest.mark.asyncio
    async def test_health_unreachable(self):
        """Health check returns ERROR status when server is unreachable."""
        client = RerankerClient(
            RerankerClientConfig(base_url="http://127.0.0.1:19999")
        )
        result = await client.health()
        assert result.status == ServiceStatus.ERROR

    @pytest.mark.asyncio
    async def test_health_success(self):
        """Health check parses response correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "reranker",
            "status": "running",
            "endpoint": "http://0.0.0.0:42983",
            "devices": ["sglang:ok"],
            "pending_requests": 0,
            "checked_at": "2026-01-01T00:00:00Z",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch(
            "services.reranker.baseline.client.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client_cls.return_value.__aexit__.return_value = False

            client = RerankerClient()
            result = await client.health()

        assert result.status == ServiceStatus.RUNNING
        assert result.devices == ["sglang:ok"]


class TestRerankerClientRetry:
    @pytest.mark.asyncio
    async def test_retry_config(self):
        """Client respects retry configuration."""
        cfg = RerankerClientConfig(retry_count=2, retry_interval=0.01)
        client = RerankerClient(cfg)
        assert client._config.retry_count == 2
        assert client._config.retry_interval == 0.01


class TestRerankerClientRank:
    @pytest.mark.asyncio
    async def test_rank_success(self):
        """Rank call parses RerankResult correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "scores": [0.95, 0.02],
            "model": "Qwen/Qwen3-32B",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "services.reranker.baseline.client.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client_cls.return_value.__aexit__.return_value = False

            client = RerankerClient()
            result = await client.rank(
                query="optimize matrix multiply",
                documents=["Use tiling for cache locality", "Hello world"],
            )

        assert isinstance(result, RerankResult)
        assert result.scores == [0.95, 0.02]
        assert result.model == "Qwen/Qwen3-32B"

    @pytest.mark.asyncio
    async def test_rank_with_instruction(self):
        """Rank passes instruction in payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"scores": [0.8], "model": "m"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "services.reranker.baseline.client.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client_cls.return_value.__aexit__.return_value = False

            client = RerankerClient()
            await client.rank(
                query="q",
                documents=["d"],
                instruction="custom instruction",
            )

        # Check the payload included instruction
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["instruction"] == "custom instruction"

    @pytest.mark.asyncio
    async def test_rank_empty_documents(self):
        """Rank with empty documents list returns empty scores."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"scores": [], "model": "m"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "services.reranker.baseline.client.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client_cls.return_value.__aexit__.return_value = False

            client = RerankerClient()
            result = await client.rank(query="q", documents=[])

        assert result.scores == []
