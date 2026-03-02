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
        assert "Experiences:" in result.output
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

    def test_cards_list_custom_type(self, tmp_path):
        """Any card_type string is accepted (free-form)."""
        result = runner.invoke(app, [
            "cards", "list",
            "--type", "custom_type",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 0

    def test_cards_list_valid_type(self, tmp_path):
        result = runner.invoke(app, [
            "cards", "list",
            "--type", "knowledge",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 0


class TestExperiencesList:
    def test_experiences_list_empty(self, tmp_path):
        result = runner.invoke(app, [
            "experiences", "list",
            "--env", "test_cli",
            "--config", str(tmp_path / "nonexistent.toml"),
        ])
        assert result.exit_code == 0
        assert "No experiences found" in result.output


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

    def test_experiences_help(self):
        result = runner.invoke(app, ["experiences", "--help"])
        assert result.exit_code == 0

    def test_orchestrate_help(self):
        result = runner.invoke(app, ["orchestrate", "--help"])
        assert result.exit_code == 0
        assert "managed subprocesses" in result.output.lower()
