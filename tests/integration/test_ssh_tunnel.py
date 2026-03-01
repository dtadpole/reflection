"""Integration test: SSH tunnel service on macOS (launchd).

Tests the full lifecycle of SSH tunnels: start, status, restart, stop.
Uses the real tunnel config from config/tunnels.yaml and exercises
launchd plist management against the _one endpoint.

Requires:
- macOS (launchd)
- SSH access to _one (centos@1and1:41922)

Run with:
    uv run pytest tests/integration/test_ssh_tunnel.py -v -s
"""

from __future__ import annotations

import plistlib
import sys
import time

import pytest

from agenix.config import PortForward, TunnelEndpoint, load_config
from services.ssh_tunnel.tunnel import LABEL_PREFIX, PLIST_DIR, check_port, get_manager

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="launchd tests only run on macOS"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def tunnel_one(config) -> TunnelEndpoint:
    for t in config.tunnels.tunnels:
        if t.name == "_one":
            return t
    pytest.skip("Tunnel _one not configured in tunnels.yaml")


@pytest.fixture(scope="module")
def manager():
    return get_manager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_port(port: int, timeout: float = 10.0) -> bool:
    """Poll until port is reachable or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_port(port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _wait_for_port_closed(port: int, timeout: float = 10.0) -> bool:
    """Poll until port is no longer reachable or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not check_port(port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


# ===========================================================================
# 1. Manager construction
# ===========================================================================


class TestManagerConstruction:
    def test_get_manager_returns_launchd(self, manager):
        """get_manager() should return LaunchdTunnelManager on macOS."""
        from services.ssh_tunnel.mac.manager import LaunchdTunnelManager

        assert isinstance(manager, LaunchdTunnelManager)

    def test_label_format(self, manager, tunnel_one):
        """Label should follow com.reflection.tunnel.<name> pattern."""
        label = manager._label(tunnel_one)
        assert label == f"{LABEL_PREFIX}.{tunnel_one.name}"

    def test_plist_path(self, manager, tunnel_one):
        """Plist should live in ~/Library/LaunchAgents/."""
        path = manager._plist_path(tunnel_one)
        assert path.parent == PLIST_DIR
        assert path.suffix == ".plist"


# ===========================================================================
# 2. Plist generation
# ===========================================================================


class TestPlistGeneration:
    def test_plist_structure(self, manager, tunnel_one):
        """Generated plist should have required launchd keys."""
        plist = manager._build_plist(tunnel_one)
        assert plist["Label"] == f"{LABEL_PREFIX}.{tunnel_one.name}"
        assert plist["KeepAlive"] is True
        assert plist["RunAtLoad"] is True
        assert "StandardErrorPath" in plist
        assert isinstance(plist["ProgramArguments"], list)

    def test_plist_ssh_args(self, manager, tunnel_one):
        """SSH args should include -N, port, -L forwards, and target."""
        args = manager._build_ssh_args(tunnel_one)
        assert "/usr/bin/ssh" in args
        assert "-N" in args
        assert "-p" in args
        assert str(tunnel_one.ssh_port) in args

        # Check -L forward args
        for fwd in tunnel_one.forwards:
            expected = f"{fwd.local_port}:{fwd.remote_host}:{fwd.remote_port}"
            assert expected in args, f"Missing forward {expected} in {args}"

        # Check target
        expected_target = (
            f"{tunnel_one.user}@{tunnel_one.host}"
            if tunnel_one.user
            else tunnel_one.host
        )
        assert expected_target in args

    def test_plist_ssh_keepalive_options(self, manager, tunnel_one):
        """SSH args should include keepalive and forward-failure options."""
        args = manager._build_ssh_args(tunnel_one)
        assert "ServerAliveInterval=30" in args
        assert "ServerAliveCountMax=3" in args
        assert "ExitOnForwardFailure=yes" in args

    def test_plist_multiple_forwards(self, manager):
        """Multiple forwards should generate multiple -L args."""
        tunnel = TunnelEndpoint(
            name="_test_multi",
            host="example.com",
            ssh_port=22,
            user="user",
            forwards=[
                PortForward(local_port=10001, remote_port=20001),
                PortForward(local_port=10002, remote_port=20002),
            ],
        )
        args = manager._build_ssh_args(tunnel)
        l_args = [a for a in args if ":" in a and a[0].isdigit()]
        assert len(l_args) == 2


# ===========================================================================
# 3. Tunnel lifecycle (start → status → stop)
# ===========================================================================


class TestTunnelLifecycle:
    """Full lifecycle test: start, verify running, stop, verify stopped.

    Tests are ordered and run sequentially within this class.
    The tunnel is started at the beginning and stopped at the end.
    """

    def test_01_stop_clean_slate(self, manager, tunnel_one):
        """Ensure tunnel is stopped before we begin (clean slate)."""
        manager.stop(tunnel_one)
        plist_path = manager._plist_path(tunnel_one)
        assert not plist_path.exists(), "Plist should be removed after stop"

        # Kill any rogue ssh processes holding our ports (e.g. manual ssh -f -N)
        import subprocess as sp

        for fwd in tunnel_one.forwards:
            sp.run(
                ["lsof", "-ti", f":{fwd.local_port}"],
                capture_output=True, text=True,
            )
            result = sp.run(
                ["lsof", "-ti", f":{fwd.local_port}"],
                capture_output=True, text=True,
            )
            for pid_str in result.stdout.strip().splitlines():
                if pid_str.isdigit():
                    import os
                    import signal

                    try:
                        os.kill(int(pid_str), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            _wait_for_port_closed(fwd.local_port, timeout=5.0)

    def test_02_status_when_stopped(self, manager, tunnel_one):
        """Status should show not running when tunnel is stopped."""
        st = manager.status(tunnel_one)
        assert st.name == tunnel_one.name
        assert st.running is False
        assert st.pid is None

    def test_03_start(self, manager, tunnel_one):
        """Start should create plist and load into launchd."""
        manager.start(tunnel_one)

        # Plist file should exist
        plist_path = manager._plist_path(tunnel_one)
        assert plist_path.exists(), "Plist file should be created"

        # Plist content should be valid
        with open(plist_path, "rb") as f:
            plist_data = plistlib.load(f)
        assert plist_data["Label"] == manager._label(tunnel_one)

    def test_04_ports_reachable(self, manager, tunnel_one):
        """After start, forwarded ports should become reachable."""
        for fwd in tunnel_one.forwards:
            ok = _wait_for_port(fwd.local_port, timeout=15.0)
            assert ok, (
                f"Port {fwd.local_port} not reachable after start. "
                f"Check SSH access to {tunnel_one.user}@{tunnel_one.host}:{tunnel_one.ssh_port}"
            )

    def test_05_status_when_running(self, manager, tunnel_one):
        """Status should show running with a PID after start."""
        st = manager.status(tunnel_one)
        assert st.running is True, f"Tunnel should be running, got: {st}"
        assert st.pid is not None, f"Should have a PID, got: {st}"
        assert st.name == tunnel_one.name
        assert len(st.forwards) == len(tunnel_one.forwards)

    def test_06_config_matches(self, manager, tunnel_one):
        """On-disk plist should match current config."""
        assert manager._config_matches(tunnel_one)

    def test_07_restart(self, manager, tunnel_one):
        """Restart should stop and re-start the tunnel."""
        manager.restart(tunnel_one)

        # Should still be reachable after restart
        for fwd in tunnel_one.forwards:
            ok = _wait_for_port(fwd.local_port, timeout=15.0)
            assert ok, f"Port {fwd.local_port} not reachable after restart"

        st = manager.status(tunnel_one)
        assert st.running is True

    def test_08_stop(self, manager, tunnel_one):
        """Stop should unload from launchd and remove plist."""
        manager.stop(tunnel_one)

        plist_path = manager._plist_path(tunnel_one)
        assert not plist_path.exists(), "Plist should be removed after stop"

    def test_09_ports_closed_after_stop(self, manager, tunnel_one):
        """Ports should become unreachable after stop."""
        # launchd may take a moment to fully terminate the ssh process
        for fwd in tunnel_one.forwards:
            closed = _wait_for_port_closed(fwd.local_port, timeout=15.0)
            assert closed, (
                f"Port {fwd.local_port} still reachable after stop. "
                "The ssh process may not have terminated yet."
            )

    def test_10_status_after_stop(self, manager, tunnel_one):
        """Status should show stopped after stop."""
        st = manager.status(tunnel_one)
        assert st.running is False


# ===========================================================================
# 4. start_all / stop_all
# ===========================================================================


class TestBulkOperations:
    """Test start_all and stop_all with real tunnels."""

    def test_01_stop_all_clean(self, manager, config):
        """Stop all tunnels for a clean starting state."""
        manager.stop_all(config.tunnels.tunnels)
        for t in config.tunnels.tunnels:
            st = manager.status(t)
            assert st.running is False

    def test_02_start_all(self, manager, config):
        """start_all should bring up all configured tunnels."""
        manager.start_all(config.tunnels.tunnels)

        for t in config.tunnels.tunnels:
            for fwd in t.forwards:
                ok = _wait_for_port(fwd.local_port, timeout=15.0)
                assert ok, f"Port {fwd.local_port} ({t.name}) not reachable after start_all"

    def test_03_start_all_idempotent(self, manager, config):
        """Calling start_all again when config hasn't changed should be a no-op."""
        # Capture PIDs before
        pids_before = {}
        for t in config.tunnels.tunnels:
            st = manager.status(t)
            pids_before[t.name] = st.pid

        manager.start_all(config.tunnels.tunnels)

        # PIDs should remain the same (no restart)
        for t in config.tunnels.tunnels:
            st = manager.status(t)
            assert st.pid == pids_before[t.name], (
                f"Tunnel {t.name} was restarted unnecessarily: "
                f"pid {pids_before[t.name]} -> {st.pid}"
            )

    def test_04_all_tunnels_running(self, manager, config):
        """All tunnels should report running status."""
        for t in config.tunnels.tunnels:
            st = manager.status(t)
            assert st.running is True, f"Tunnel {t.name} not running"

    # NOTE: we intentionally leave tunnels running after this test class
    # so the subsequent kbEval service tests can use them.


# ===========================================================================
# 5. check_port utility
# ===========================================================================


class TestCheckPort:
    """Test the check_port utility function."""

    def test_unreachable_port(self):
        """A port with nothing listening should return False."""
        assert check_port(19999) is False

    def test_reachable_port_after_start(self, manager, config):
        """Ports should be reachable when tunnels are running."""
        # Tunnels should be running from TestBulkOperations
        for t in config.tunnels.tunnels:
            for fwd in t.forwards:
                assert check_port(fwd.local_port), (
                    f"Port {fwd.local_port} ({t.name}) should be reachable"
                )
