"""Launchd-based SSH tunnel manager for macOS."""

from __future__ import annotations

import plistlib
import subprocess

from agenix.config import TunnelEndpoint
from services.ssh_tunnel.tunnel import (
    LABEL_PREFIX,
    PLIST_DIR,
    TunnelStatus,
    check_port,
)


class LaunchdTunnelManager:
    """Manage SSH tunnels via macOS launchd plist files."""

    def _label(self, tunnel: TunnelEndpoint) -> str:
        return f"{LABEL_PREFIX}.{tunnel.name}"

    def _plist_path(self, tunnel: TunnelEndpoint):
        return PLIST_DIR / f"{self._label(tunnel)}.plist"

    def _build_ssh_args(self, tunnel: TunnelEndpoint) -> list[str]:
        args = [
            "/usr/bin/ssh", "-N",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-p", str(tunnel.ssh_port),
        ]
        for fwd in tunnel.forwards:
            args.extend([
                "-L",
                f"{fwd.local_port}:{fwd.remote_host}:{fwd.remote_port}",
            ])
        target = f"{tunnel.user}@{tunnel.host}" if tunnel.user else tunnel.host
        args.append(target)
        return args

    def _build_plist(self, tunnel: TunnelEndpoint) -> dict:
        label = self._label(tunnel)
        return {
            "Label": label,
            "ProgramArguments": self._build_ssh_args(tunnel),
            "KeepAlive": True,
            "RunAtLoad": True,
            "StandardErrorPath": f"/tmp/reflection-tunnel-{tunnel.name}.err",
        }

    def start(self, tunnel: TunnelEndpoint) -> None:
        """Write plist and load into launchd."""
        PLIST_DIR.mkdir(parents=True, exist_ok=True)
        plist_path = self._plist_path(tunnel)
        plist_data = self._build_plist(tunnel)
        with open(plist_path, "wb") as f:
            plistlib.dump(plist_data, f)
        subprocess.run(
            ["launchctl", "load", str(plist_path)],
            check=True,
            capture_output=True,
        )

    def stop(self, tunnel: TunnelEndpoint) -> None:
        """Unload from launchd and remove plist."""
        import os
        import signal

        # Capture PID before unload so we can ensure cleanup
        st = self.status(tunnel)
        pid = st.pid

        plist_path = self._plist_path(tunnel)
        if plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                capture_output=True,
            )
            plist_path.unlink(missing_ok=True)

        # Ensure the ssh process is terminated (launchctl unload may be async)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def restart(self, tunnel: TunnelEndpoint) -> None:
        """Stop and re-start a tunnel (useful when forwards change)."""
        self.stop(tunnel)
        self.start(tunnel)

    def status(self, tunnel: TunnelEndpoint) -> TunnelStatus:
        """Check launchd job status and port reachability."""
        label = self._label(tunnel)
        pid = None
        loaded = False

        result = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            loaded = True
            # launchctl list <label> outputs plist-like dict with "PID" = <num>;
            import re

            pid_match = re.search(r'"PID"\s*=\s*(\d+)', result.stdout)
            if pid_match:
                pid = int(pid_match.group(1))

        # Verify ports are actually forwarding traffic
        running = loaded and all(
            check_port(fwd.local_port) for fwd in tunnel.forwards
        )

        return TunnelStatus(
            name=tunnel.name,
            running=running,
            forwards=list(tunnel.forwards),
            pid=pid,
        )

    def _config_matches(self, tunnel: TunnelEndpoint) -> bool:
        """Check if the on-disk plist matches the current tunnel config."""
        plist_path = self._plist_path(tunnel)
        if not plist_path.exists():
            return False
        with open(plist_path, "rb") as f:
            existing = plistlib.load(f)
        expected = self._build_plist(tunnel)
        return existing == expected

    def start_all(self, tunnels: list[TunnelEndpoint]) -> None:
        for t in tunnels:
            if not self._config_matches(t):
                self.restart(t)
            else:
                st = self.status(t)
                if not st.running:
                    self.start(t)

    def stop_all(self, tunnels: list[TunnelEndpoint]) -> None:
        for t in tunnels:
            self.stop(t)

    def restart_all(self, tunnels: list[TunnelEndpoint]) -> None:
        for t in tunnels:
            self.restart(t)
