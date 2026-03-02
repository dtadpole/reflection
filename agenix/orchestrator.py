"""Orchestrator: multi-agent process manager.

Spawns and manages all agent processes according to configuration.
One-shot agents (curator) run to completion first, then long-running
agents are spawned in parallel and monitored until shutdown.
"""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from agenix.config import AgentSpec, OrchestratorConfig, make_log_path

log = logging.getLogger(__name__)

ONE_SHOT_AGENTS = {"curator"}

# Map config names (underscore) to CLI names (hyphen)
_CLI_NAME_MAP = {
    "insight_finder": "insight-finder",
}


@dataclass
class ManagedProcess:
    agent_name: str
    instance: int
    proc: subprocess.Popen
    started_at: float
    log_path: Path | None = None
    _log_fh: object = field(default=None, repr=False)
    label: str = field(init=False)

    def __post_init__(self):
        self.label = f"{self.agent_name}[{self.instance}]"

    def close_log(self):
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    @property
    def exit_code(self) -> int | None:
        return self.proc.poll()


class Orchestrator:
    """Spawns agent subprocesses and manages their lifecycle."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        env: str | None = None,
        config_path: str | None = None,
        verbose: bool = False,
        logs_dir: Path | None = None,
    ):
        self._config = config
        self._env = env
        self._config_path = config_path
        self._verbose = verbose
        self._logs_dir = logs_dir
        self._children: list[ManagedProcess] = []
        self._shutdown = False
        self._seq_counters: dict[str, int] = {}
        self._reflection_bin = self._find_reflection_bin()

    def run(self) -> int:
        """Main entry point. Returns exit code (0 = clean shutdown)."""
        prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        try:
            return self._run_inner()
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

    def _run_inner(self) -> int:
        one_shot = [s for s in self._config.agents if s.name in ONE_SHOT_AGENTS]
        long_running = [s for s in self._config.agents if s.name not in ONE_SHOT_AGENTS]

        # Phase 1: run one-shot agents to completion
        if one_shot and not self._shutdown:
            log.info("Running %d one-shot agent(s)...", len(one_shot))
            exit_code = self._run_one_shot(one_shot)
            if exit_code != 0:
                log.error("One-shot agents failed (exit_code=%d)", exit_code)
                return exit_code

        if self._shutdown:
            return 0

        # Phase 2: spawn long-running agents
        if long_running:
            self._spawn_long_running(long_running)

        if not self._children:
            log.info("No long-running agents to manage.")
            return 0

        # Phase 3: monitor loop
        self._monitor_loop()

        # Phase 4: shutdown
        self._shutdown_children()

        # Return non-zero if any child crashed
        crashed = [c for c in self._children if c.exit_code not in (None, 0)]
        return 1 if crashed else 0

    def _run_one_shot(self, specs: list[AgentSpec]) -> int:
        """Run one-shot agents sequentially, wait for each to complete."""
        for spec in specs:
            if self._shutdown:
                return 0
            for i in range(spec.count):
                if self._shutdown:
                    return 0
                cmd = self._build_command(spec)
                seq_id = self._next_seq_id(spec.name)
                label = f"{spec.name}[{seq_id}]"
                log_path, log_fh = self._open_log_file(spec.name, seq_id)
                if log_path:
                    log.info("Starting one-shot agent %s (log=%s)", label, log_path)
                else:
                    log.info("Starting one-shot agent %s: %s", label, " ".join(cmd))
                try:
                    stdout_target = log_fh if log_fh else (
                        None if self._verbose else subprocess.DEVNULL
                    )
                    proc = subprocess.Popen(
                        cmd,
                        stdout=stdout_target,
                        stderr=subprocess.STDOUT if log_fh else stdout_target,
                    )
                    proc.wait()
                    if log_fh:
                        log_fh.close()
                    if proc.returncode != 0:
                        log.error("%s exited with code %d", label, proc.returncode)
                        return proc.returncode
                    log.info("%s completed successfully.", label)
                except Exception as e:
                    if log_fh:
                        log_fh.close()
                    log.error("Failed to start %s: %s", label, e)
                    return 1
        return 0

    def _spawn_long_running(self, specs: list[AgentSpec]) -> None:
        """Spawn long-running agent processes in parallel."""
        for spec in specs:
            for i in range(spec.count):
                cmd = self._build_command(spec)
                seq_id = self._next_seq_id(spec.name)
                label = f"{spec.name}[{seq_id}]"
                log_path, log_fh = self._open_log_file(spec.name, seq_id)
                if log_path:
                    log.info("Spawning %s (log=%s)", label, log_path)
                else:
                    log.info("Spawning %s: %s", label, " ".join(cmd))
                try:
                    stdout_target = log_fh if log_fh else (
                        None if self._verbose else subprocess.DEVNULL
                    )
                    proc = subprocess.Popen(
                        cmd,
                        stdout=stdout_target,
                        stderr=subprocess.STDOUT if log_fh else stdout_target,
                    )
                    managed = ManagedProcess(
                        agent_name=spec.name,
                        instance=seq_id,
                        proc=proc,
                        started_at=time.monotonic(),
                        log_path=log_path,
                        _log_fh=log_fh,
                    )
                    self._children.append(managed)
                    log.info("%s spawned (pid=%d)", label, proc.pid)
                except Exception as e:
                    if log_fh:
                        log_fh.close()
                    log.error("Failed to spawn %s: %s", label, e)

    def _monitor_loop(self) -> None:
        """Poll children until shutdown or all children exit."""
        last_status = time.monotonic()

        while not self._shutdown:
            # Check if all children have exited
            if all(not c.running for c in self._children):
                log.info("All children have exited.")
                break

            # Report status periodically
            now = time.monotonic()
            if now - last_status >= self._config.status_interval:
                self._report_status()
                last_status = now

            time.sleep(2)

    def _shutdown_children(self) -> None:
        """Wait for children to exit gracefully, then force-kill stragglers."""
        running = [c for c in self._children if c.running]
        if not running:
            return

        log.info(
            "Waiting up to %ds for %d child(ren) to exit...",
            self._config.shutdown_timeout,
            len(running),
        )

        deadline = time.monotonic() + self._config.shutdown_timeout
        while time.monotonic() < deadline:
            running = [c for c in self._children if c.running]
            if not running:
                log.info("All children exited gracefully.")
                return
            time.sleep(0.5)

        # Force-kill stragglers
        stragglers = [c for c in self._children if c.running]
        if stragglers:
            log.warning(
                "Force-killing %d straggler(s): %s",
                len(stragglers),
                ", ".join(c.label for c in stragglers),
            )
            for c in stragglers:
                try:
                    c.proc.kill()
                except OSError:
                    pass

        # Close log file handles
        for c in self._children:
            c.close_log()

    def _next_seq_id(self, agent_name: str) -> int:
        """Return the next sequential ID for the given agent name."""
        seq = self._seq_counters.get(agent_name, 0)
        self._seq_counters[agent_name] = seq + 1
        return seq

    def _open_log_file(self, agent_name: str, seq_id: int):
        """Create a log file and return (path, file_handle). Returns (None, None) if no logs_dir."""
        if self._logs_dir is None:
            return None, None
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        path = make_log_path(self._logs_dir, agent_name, seq_id)
        fh = open(path, "w")  # noqa: SIM115
        return path, fh

    def _build_command(self, spec: AgentSpec) -> list[str]:
        """Build the subprocess command for an agent spec."""
        cli_name = _CLI_NAME_MAP.get(spec.name, spec.name)
        cmd = [str(self._reflection_bin), "agent", cli_name]

        if self._env:
            cmd.extend(["--env", self._env])
        if self._config_path:
            cmd.extend(["--config", self._config_path])
        if self._verbose:
            cmd.append("--verbose")

        for key, value in spec.options.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        return cmd

    def _report_status(self) -> None:
        """Print status of all managed processes."""
        for c in self._children:
            if c.running:
                elapsed = time.monotonic() - c.started_at
                log.info("%s: running (pid=%d, elapsed=%.0fs)", c.label, c.proc.pid, elapsed)
            else:
                log.info("%s: exited (code=%s)", c.label, c.exit_code)

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        log.info("Received %s, initiating shutdown...", sig_name)
        self._shutdown = True

    @staticmethod
    def _find_reflection_bin() -> Path:
        """Find the reflection CLI binary."""
        # Prefer sibling of current Python executable (uv-managed venv)
        candidate = Path(sys.executable).parent / "reflection"
        if candidate.exists():
            return candidate
        # Fallback to PATH lookup
        found = shutil.which("reflection")
        if found:
            return Path(found)
        raise FileNotFoundError(
            "Cannot find 'reflection' CLI binary. "
            "Ensure the package is installed (uv sync)."
        )
