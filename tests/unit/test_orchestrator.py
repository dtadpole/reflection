"""Tests for the orchestrator process manager."""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from agenix.config import AgentSpec, OrchestratorConfig, make_log_path
from agenix.orchestrator import _CLI_NAME_MAP, ManagedProcess, Orchestrator


class TestAgentSpec:
    def test_defaults(self):
        spec = AgentSpec(name="solver")
        assert spec.name == "solver"
        assert spec.count == 1
        assert spec.options == {}

    def test_with_options(self):
        spec = AgentSpec(name="curator", count=1, options={"n": 100, "levels": "level_1"})
        assert spec.options["n"] == 100
        assert spec.options["levels"] == "level_1"

    def test_count_override(self):
        spec = AgentSpec(name="solver", count=3)
        assert spec.count == 3


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.shutdown_timeout == 30
        assert cfg.status_interval == 60
        assert len(cfg.agents) == 5
        names = [a.name for a in cfg.agents]
        assert "curator" in names
        assert "solver" in names
        assert "critic" in names
        assert "organizer" in names
        assert "insight_finder" in names

    def test_solver_default_count(self):
        cfg = OrchestratorConfig()
        solver = next(a for a in cfg.agents if a.name == "solver")
        assert solver.count == 2

    def test_custom_config(self):
        cfg = OrchestratorConfig(
            agents=[AgentSpec(name="solver", count=4)],
            shutdown_timeout=10,
            status_interval=30,
        )
        assert len(cfg.agents) == 1
        assert cfg.agents[0].count == 4
        assert cfg.shutdown_timeout == 10
        assert cfg.status_interval == 30


class TestManagedProcess:
    def _make_proc(self, poll_value=None) -> ManagedProcess:
        mock_popen = MagicMock(spec=subprocess.Popen)
        mock_popen.poll.return_value = poll_value
        mock_popen.pid = 12345
        return ManagedProcess(
            agent_name="solver",
            instance=0,
            proc=mock_popen,
            started_at=time.monotonic(),
        )

    def test_label(self):
        mp = self._make_proc()
        assert mp.label == "solver[0]"

    def test_label_instance(self):
        mock_popen = MagicMock(spec=subprocess.Popen)
        mock_popen.poll.return_value = None
        mp = ManagedProcess(
            agent_name="solver", instance=2, proc=mock_popen, started_at=0.0
        )
        assert mp.label == "solver[2]"

    def test_running_true(self):
        mp = self._make_proc(poll_value=None)
        assert mp.running is True

    def test_running_false(self):
        mp = self._make_proc(poll_value=0)
        assert mp.running is False

    def test_exit_code_none_while_running(self):
        mp = self._make_proc(poll_value=None)
        assert mp.exit_code is None

    def test_exit_code_zero(self):
        mp = self._make_proc(poll_value=0)
        assert mp.exit_code == 0

    def test_exit_code_nonzero(self):
        mp = self._make_proc(poll_value=1)
        assert mp.exit_code == 1


class TestBuildCommand:
    @patch.object(Orchestrator, "_find_reflection_bin")
    def _make_orch(self, mock_find, **kwargs):
        mock_find.return_value = "/usr/bin/reflection"
        defaults = dict(env=None, config_path=None, verbose=False)
        defaults.update(kwargs)
        return Orchestrator(
            OrchestratorConfig(agents=[]),
            **defaults,
        )

    def test_basic_command(self):
        orch = self._make_orch()
        spec = AgentSpec(name="solver")
        cmd = orch._build_command(spec)
        assert cmd == ["/usr/bin/reflection", "agent", "solver"]

    def test_insight_finder_hyphen(self):
        orch = self._make_orch()
        spec = AgentSpec(name="insight_finder")
        cmd = orch._build_command(spec)
        assert cmd[2] == "insight-finder"

    def test_with_env(self):
        orch = self._make_orch(env="prod")
        spec = AgentSpec(name="critic")
        cmd = orch._build_command(spec)
        assert "--env" in cmd
        idx = cmd.index("--env")
        assert cmd[idx + 1] == "prod"

    def test_with_config_path(self):
        orch = self._make_orch(config_path="/tmp/config.toml")
        spec = AgentSpec(name="solver")
        cmd = orch._build_command(spec)
        assert "--config" in cmd
        idx = cmd.index("--config")
        assert cmd[idx + 1] == "/tmp/config.toml"

    def test_with_verbose(self):
        orch = self._make_orch(verbose=True)
        spec = AgentSpec(name="solver")
        cmd = orch._build_command(spec)
        assert "--verbose" in cmd

    def test_with_int_option(self):
        orch = self._make_orch()
        spec = AgentSpec(name="organizer", options={"interval": 300})
        cmd = orch._build_command(spec)
        assert "--interval" in cmd
        idx = cmd.index("--interval")
        assert cmd[idx + 1] == "300"

    def test_with_str_option(self):
        orch = self._make_orch()
        spec = AgentSpec(name="curator", options={"levels": "level_1,level_2"})
        cmd = orch._build_command(spec)
        assert "--levels" in cmd
        idx = cmd.index("--levels")
        assert cmd[idx + 1] == "level_1,level_2"

    def test_with_bool_option_true(self):
        orch = self._make_orch()
        spec = AgentSpec(name="solver", options={"some_flag": True})
        cmd = orch._build_command(spec)
        assert "--some-flag" in cmd

    def test_with_bool_option_false(self):
        orch = self._make_orch()
        spec = AgentSpec(name="solver", options={"some_flag": False})
        cmd = orch._build_command(spec)
        assert "--some-flag" not in cmd

    def test_option_underscore_to_hyphen(self):
        orch = self._make_orch()
        spec = AgentSpec(name="curator", options={"my_option": "val"})
        cmd = orch._build_command(spec)
        assert "--my-option" in cmd

    def test_full_command(self):
        orch = self._make_orch(env="test_user", config_path="/c.toml", verbose=True)
        spec = AgentSpec(name="insight_finder", options={"interval": 600})
        cmd = orch._build_command(spec)
        assert cmd[0] == "/usr/bin/reflection"
        assert cmd[1] == "agent"
        assert cmd[2] == "insight-finder"
        assert "--env" in cmd
        assert "--config" in cmd
        assert "--verbose" in cmd
        assert "--interval" in cmd


class TestShutdownChildren:
    @patch.object(Orchestrator, "_find_reflection_bin")
    def test_skips_already_exited(self, mock_find):
        mock_find.return_value = "/usr/bin/reflection"
        orch = Orchestrator(OrchestratorConfig(agents=[], shutdown_timeout=1))

        # Add already-exited child
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0
        mock_proc.pid = 111
        mp = ManagedProcess(
            agent_name="solver", instance=0, proc=mock_proc, started_at=0.0
        )
        orch._children.append(mp)

        orch._shutdown_children()
        mock_proc.kill.assert_not_called()

    @patch.object(Orchestrator, "_find_reflection_bin")
    def test_kills_straggler(self, mock_find):
        mock_find.return_value = "/usr/bin/reflection"
        orch = Orchestrator(OrchestratorConfig(agents=[], shutdown_timeout=0))

        # Add a process that never exits
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # always running
        mock_proc.pid = 222
        mp = ManagedProcess(
            agent_name="critic", instance=0, proc=mock_proc, started_at=0.0
        )
        orch._children.append(mp)

        orch._shutdown_children()
        mock_proc.kill.assert_called_once()


class TestSignalHandler:
    @patch.object(Orchestrator, "_find_reflection_bin")
    def test_sets_shutdown_flag(self, mock_find):
        mock_find.return_value = "/usr/bin/reflection"
        orch = Orchestrator(OrchestratorConfig(agents=[]))
        assert orch._shutdown is False
        orch._handle_signal(signal.SIGINT, None)
        assert orch._shutdown is True


class TestCLINameMap:
    def test_insight_finder_mapped(self):
        assert _CLI_NAME_MAP["insight_finder"] == "insight-finder"

    def test_unmapped_passthrough(self):
        assert _CLI_NAME_MAP.get("solver", "solver") == "solver"


class TestMakeLogPath:
    def test_with_seq_id(self):
        p = make_log_path(Path("/tmp/logs"), "solver", seq_id=0)
        assert p.parent == Path("/tmp/logs")
        assert p.name.startswith("solver_0_")
        assert p.name.endswith(".log")

    def test_without_seq_id(self):
        p = make_log_path(Path("/tmp/logs"), "critic")
        assert p.parent == Path("/tmp/logs")
        assert p.name.startswith("critic_")
        assert p.name.endswith(".log")
        # Should not have a seq_id field — pattern is critic_YYYYMMDD_HHMMSS.log
        parts = p.stem.split("_")  # critic, YYYYMMDD, HHMMSS
        assert len(parts) == 3

    def test_with_seq_id_parts(self):
        p = make_log_path(Path("/tmp/logs"), "solver", seq_id=2)
        parts = p.stem.split("_")  # solver, 2, YYYYMMDD, HHMMSS
        assert len(parts) == 4
        assert parts[0] == "solver"
        assert parts[1] == "2"


class TestSeqIdAssignment:
    @patch.object(Orchestrator, "_find_reflection_bin")
    def test_increments_per_agent(self, mock_find):
        mock_find.return_value = "/usr/bin/reflection"
        orch = Orchestrator(OrchestratorConfig(agents=[]))
        assert orch._next_seq_id("solver") == 0
        assert orch._next_seq_id("solver") == 1
        assert orch._next_seq_id("solver") == 2
        assert orch._next_seq_id("critic") == 0
        assert orch._next_seq_id("solver") == 3

    @patch.object(Orchestrator, "_find_reflection_bin")
    def test_independent_per_agent(self, mock_find):
        mock_find.return_value = "/usr/bin/reflection"
        orch = Orchestrator(OrchestratorConfig(agents=[]))
        assert orch._next_seq_id("solver") == 0
        assert orch._next_seq_id("critic") == 0
        assert orch._next_seq_id("organizer") == 0


class TestLogFileCreation:
    @patch.object(Orchestrator, "_find_reflection_bin")
    def test_creates_log_file(self, mock_find, tmp_path):
        mock_find.return_value = "/usr/bin/reflection"
        logs_dir = tmp_path / "logs"
        orch = Orchestrator(
            OrchestratorConfig(agents=[]),
            logs_dir=logs_dir,
        )
        path, fh = orch._open_log_file("solver", 0)
        assert path is not None
        assert fh is not None
        assert path.parent == logs_dir
        assert path.name.startswith("solver_0_")
        fh.close()

    @patch.object(Orchestrator, "_find_reflection_bin")
    def test_no_log_without_logs_dir(self, mock_find):
        mock_find.return_value = "/usr/bin/reflection"
        orch = Orchestrator(OrchestratorConfig(agents=[]))
        path, fh = orch._open_log_file("solver", 0)
        assert path is None
        assert fh is None


class TestManagedProcessLog:
    def test_close_log(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_fh = MagicMock()
        mp = ManagedProcess(
            agent_name="solver",
            instance=0,
            proc=mock_proc,
            started_at=0.0,
            log_path=Path("/tmp/solver.log"),
            _log_fh=mock_fh,
        )
        mp.close_log()
        mock_fh.close.assert_called_once()
        assert mp._log_fh is None

    def test_close_log_noop_without_fh(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mp = ManagedProcess(
            agent_name="solver",
            instance=0,
            proc=mock_proc,
            started_at=0.0,
        )
        mp.close_log()  # should not raise
