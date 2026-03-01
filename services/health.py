"""Health checker for remote service endpoints."""

from __future__ import annotations

import asyncio
import logging

from agenix.config import ServiceEndpoint, ServicesConfig
from services.kb_eval.client import KbEvalClient
from services.models import ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)


class HealthChecker:
    """Probes configured service endpoints for health status."""

    def __init__(self, config: ServicesConfig) -> None:
        self._config = config

    async def check_endpoint(self, endpoint: ServiceEndpoint) -> ServiceHealth:
        """Check health of a single endpoint's kbEval service.

        Args:
            endpoint: The service endpoint to probe.

        Returns:
            ServiceHealth with status, device info, and pending request count.
        """
        client = KbEvalClient(endpoint.kb_eval)
        try:
            return await client.health()
        except Exception as e:
            logger.debug("Health check failed for %s: %s", endpoint.name, e)
            return ServiceHealth(
                name=endpoint.name,
                status=ServiceStatus.ERROR,
                endpoint=endpoint.kb_eval.base_url,
            )

    async def check_all(self) -> list[ServiceHealth]:
        """Check health of all configured endpoints concurrently.

        Returns:
            List of ServiceHealth results, one per endpoint.
        """
        if not self._config.endpoints:
            return []

        tasks = [self.check_endpoint(ep) for ep in self._config.endpoints]
        return list(await asyncio.gather(*tasks))

    async def check_ssh(self, endpoint: ServiceEndpoint) -> bool:
        """Test SSH connectivity to an endpoint.

        Args:
            endpoint: The endpoint to test SSH connectivity for.

        Returns:
            True if SSH connection succeeds.
        """
        try:
            import asyncssh

            conn = await asyncio.wait_for(
                asyncssh.connect(
                    endpoint.host,
                    port=endpoint.port,
                    username=endpoint.user or None,
                    known_hosts=None,
                ),
                timeout=10.0,
            )
            conn.close()
            return True
        except Exception as e:
            logger.debug("SSH check failed for %s: %s", endpoint.name, e)
            return False
