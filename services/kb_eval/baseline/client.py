"""Async HTTP client for the kbEval service."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from agenix.config import KbEvalClientConfig
from services.models import KernelExecResult, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)


class KbEvalClient:
    """Client for communicating with a kbEval server.

    Features:
    - Gzip request compression for large code payloads
    - Exponential backoff retry on transient failures
    - Configurable timeouts
    """

    def __init__(self, config: KbEvalClientConfig | None = None) -> None:
        self._config = config or KbEvalClientConfig()
        self._base_url = self._config.base_url.rstrip("/")

    async def eval(
        self,
        reference_code: str,
        generated_code: str,
        run_tag: str = "",
        task_tag: str = "",
        code_type: str = "triton",
    ) -> KernelExecResult:
        """Evaluate generated kernel against reference.

        Args:
            reference_code: Reference implementation source.
            generated_code: Generated kernel source.
            run_tag: Run identifier for organizing results.
            task_tag: Task identifier within the run.
            code_type: One of "triton", "cuda", "pytorch".

        Returns:
            KernelExecResult with compilation, correctness, and timing data.
        """
        payload = {
            "reference_code": reference_code,
            "generated_code": generated_code,
            "run_tag": run_tag,
            "task_tag": task_tag,
            "code_type": code_type,
        }
        data = await self._post("/eval", payload)
        return KernelExecResult.model_validate(data)

    async def eval_ref(
        self,
        reference_code: str,
        run_tag: str = "",
        task_tag: str = "",
    ) -> KernelExecResult:
        """Benchmark reference code only.

        Args:
            reference_code: Reference implementation source.
            run_tag: Run identifier.
            task_tag: Task identifier.

        Returns:
            KernelExecResult with reference timing data.
        """
        payload = {
            "reference_code": reference_code,
            "run_tag": run_tag,
            "task_tag": task_tag,
        }
        data = await self._post("/eval_ref", payload)
        return KernelExecResult.model_validate(data)

    async def health(self) -> ServiceHealth:
        """Check kbEval server health.

        Returns:
            ServiceHealth with status and device info.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
                return ServiceHealth.model_validate(resp.json())
        except Exception as e:
            logger.debug("Health check failed: %s", e)
            return ServiceHealth(
                name="kb_eval",
                status=ServiceStatus.ERROR,
                endpoint=self._base_url,
            )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON with retry."""
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}

        last_exc: Exception | None = None
        for attempt in range(self._config.retry_count):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self._config.timeout, connect=30.0)
                ) as client:
                    resp = await client.post(url, content=body, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as e:
                last_exc = e
                # Don't retry on 4xx client errors (not transient)
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
            except httpx.ReadTimeout:
                # Read timeouts mean the server is still processing — don't
                # retry as the kernel is likely too slow to evaluate.
                raise
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
