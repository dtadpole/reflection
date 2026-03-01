"""SSH tunnel management: status checks, port probing, and manager selection."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from agenix.config import PortForward, TunnelEndpoint

PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
LABEL_PREFIX = "com.reflection.tunnel"


class TunnelStatus(BaseModel):
    name: str
    running: bool
    forwards: list[PortForward]
    pid: int | None = None


class TunnelManager(Protocol):
    def start(self, tunnel: TunnelEndpoint) -> None: ...
    def stop(self, tunnel: TunnelEndpoint) -> None: ...
    def restart(self, tunnel: TunnelEndpoint) -> None: ...
    def status(self, tunnel: TunnelEndpoint) -> TunnelStatus: ...
    def start_all(self, tunnels: list[TunnelEndpoint]) -> None: ...
    def stop_all(self, tunnels: list[TunnelEndpoint]) -> None: ...
    def restart_all(self, tunnels: list[TunnelEndpoint]) -> None: ...


def check_port(port: int, host: str = "localhost", timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_manager() -> TunnelManager:
    """Return the platform-appropriate TunnelManager."""
    if sys.platform == "darwin":
        from services.ssh_tunnel.mac.manager import LaunchdTunnelManager

        return LaunchdTunnelManager()
    else:
        from services.ssh_tunnel.linux.manager import SystemdTunnelManager

        return SystemdTunnelManager()
