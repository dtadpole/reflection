"""Systemd-based SSH tunnel manager for Linux (stub)."""

from __future__ import annotations

from agenix.config import TunnelEndpoint
from services.ssh_tunnel.tunnel import TunnelStatus


class SystemdTunnelManager:
    """Placeholder for Linux systemd-based tunnel management."""

    def start(self, tunnel: TunnelEndpoint) -> None:
        raise NotImplementedError("Linux variant not yet implemented")

    def stop(self, tunnel: TunnelEndpoint) -> None:
        raise NotImplementedError("Linux variant not yet implemented")

    def restart(self, tunnel: TunnelEndpoint) -> None:
        raise NotImplementedError("Linux variant not yet implemented")

    def status(self, tunnel: TunnelEndpoint) -> TunnelStatus:
        raise NotImplementedError("Linux variant not yet implemented")

    def start_all(self, tunnels: list[TunnelEndpoint]) -> None:
        raise NotImplementedError("Linux variant not yet implemented")

    def stop_all(self, tunnels: list[TunnelEndpoint]) -> None:
        raise NotImplementedError("Linux variant not yet implemented")

    def restart_all(self, tunnels: list[TunnelEndpoint]) -> None:
        raise NotImplementedError("Linux variant not yet implemented")
