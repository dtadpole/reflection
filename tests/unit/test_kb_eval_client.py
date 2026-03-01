"""Tests for kbEval HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenix.config import KbEvalClientConfig
from services.kb_eval.baseline.client import KbEvalClient
from services.models import ServiceStatus


class TestKbEvalClientConstruction:
    def test_default_config(self):
        client = KbEvalClient()
        assert client._base_url == "http://localhost:8456"

    def test_custom_config(self):
        cfg = KbEvalClientConfig(base_url="http://gpu1:9000", timeout=60)
        client = KbEvalClient(cfg)
        assert client._base_url == "http://gpu1:9000"
        assert client._config.timeout == 60

    def test_trailing_slash_stripped(self):
        cfg = KbEvalClientConfig(base_url="http://gpu1:8456/")
        client = KbEvalClient(cfg)
        assert client._base_url == "http://gpu1:8456"


class TestKbEvalClientHealth:
    @pytest.mark.asyncio
    async def test_health_unreachable(self):
        """Health check returns ERROR status when server is unreachable."""
        client = KbEvalClient(KbEvalClientConfig(base_url="http://127.0.0.1:19999"))
        result = await client.health()
        assert result.status == ServiceStatus.ERROR

    @pytest.mark.asyncio
    async def test_health_success(self):
        """Health check parses response correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "kb_eval",
            "status": "running",
            "endpoint": "http://gpu1:8456",
            "devices": ["cuda:0"],
            "pending_requests": 0,
            "checked_at": "2026-01-01T00:00:00Z",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch("services.kb_eval.baseline.client.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client_cls.return_value.__aexit__.return_value = False

            client = KbEvalClient()
            result = await client.health()

        assert result.status == ServiceStatus.RUNNING
        assert result.devices == ["cuda:0"]


class TestKbEvalClientRetry:
    @pytest.mark.asyncio
    async def test_retry_config(self):
        """Client respects retry configuration."""
        cfg = KbEvalClientConfig(retry_count=2, retry_interval=0.01)
        client = KbEvalClient(cfg)
        assert client._config.retry_count == 2
        assert client._config.retry_interval == 0.01
