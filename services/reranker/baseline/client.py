"""Async HTTP client for the reranker service."""

from __future__ import annotations

import logging

import httpx

from agenix.config import RerankerClientConfig
from services.models import RerankResult, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)


class RerankerClient:
    """Client for communicating with a reranker server."""

    def __init__(self, config: RerankerClientConfig | None = None) -> None:
        self._config = config or RerankerClientConfig()
        self._base_url = self._config.base_url.rstrip("/")

    async def rank(
        self,
        query: str,
        documents: list[str],
        instruction: str = "",
        top_k: int = 0,
    ) -> RerankResult:
        """Rerank documents by relevance to a query.

        Args:
            query: The search query.
            documents: Documents to rerank.
            instruction: Optional instruction for relevance judgement.
            top_k: Return only top-K results (0 = all).

        Returns:
            RerankResult with per-document scores.
        """
        payload: dict = {"query": query, "documents": documents}
        if instruction:
            payload["instruction"] = instruction
        if top_k:
            payload["top_k"] = top_k
        data = await self._post("/rank", payload)
        return RerankResult.model_validate(data)

    async def health(self) -> ServiceHealth:
        """Check reranker server health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
                return ServiceHealth.model_validate(resp.json())
        except Exception as e:
            logger.debug("Health check failed: %s", e)
            return ServiceHealth(
                name="reranker",
                status=ServiceStatus.ERROR,
                endpoint=self._base_url,
            )

    async def _post(self, path: str, payload: dict) -> dict:
        """POST JSON with retry."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self._config.retry_count):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self._config.timeout, connect=30.0)
                ) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as e:
                last_exc = e
                if 400 <= e.response.status_code < 500:
                    break
                if attempt < self._config.retry_count - 1:
                    wait = self._config.retry_interval ** (attempt + 1)
                    logger.warning(
                        "Request to %s failed (attempt %d/%d), retrying in %.1fs: %s",
                        url, attempt + 1, self._config.retry_count, wait, e,
                    )
                    import asyncio

                    await asyncio.sleep(wait)
            except httpx.TransportError as e:
                last_exc = e
                if attempt < self._config.retry_count - 1:
                    wait = self._config.retry_interval ** (attempt + 1)
                    logger.warning(
                        "Request to %s failed (attempt %d/%d), retrying in %.1fs: %s",
                        url, attempt + 1, self._config.retry_count, wait, e,
                    )
                    import asyncio

                    await asyncio.sleep(wait)

        raise ConnectionError(
            f"All {self._config.retry_count} attempts to {url} failed"
        ) from last_exc
