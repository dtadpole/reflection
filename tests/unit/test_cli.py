"""Tests for the CLI entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


class TestStatus:
    def test_status_default(self, tmp_path):
        """Status on empty data should show zero counts."""
        result = runner.invoke(app, [
            "status",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 0
        assert "Problems:" in result.output
        assert "Trajectories:" in result.output
        assert "Cards:" in result.output

    def test_status_verbose(self, tmp_path):
        result = runner.invoke(app, [
            "status",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
            "--verbose",
        ])
        assert result.exit_code == 0


class TestCardsList:
    def test_cards_list_empty(self, tmp_path):
        result = runner.invoke(app, [
            "cards", "list",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 0
        assert "No cards found" in result.output

    def test_cards_list_invalid_type(self, tmp_path):
        result = runner.invoke(app, [
            "cards", "list",
            "--type", "invalid",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 1

    def test_cards_list_valid_type(self, tmp_path):
        result = runner.invoke(app, [
            "cards", "list",
            "--type", "knowledge",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 0


class TestTrajectoriesList:
    def test_trajectories_list_empty(self, tmp_path):
        result = runner.invoke(app, [
            "trajectories", "list",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 0
        assert "No trajectories found" in result.output


class TestHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "reflection" in result.output.lower() or "Usage" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "iterations" in result.output.lower()

    def test_solve_help(self):
        result = runner.invoke(app, ["solve", "--help"])
        assert result.exit_code == 0

    def test_cards_help(self):
        result = runner.invoke(app, ["cards", "--help"])
        assert result.exit_code == 0

    def test_trajectories_help(self):
        result = runner.invoke(app, ["trajectories", "--help"])
        assert result.exit_code == 0
