"""Smoke tests for services CLI commands."""

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


class TestServicesStatus:
    def test_status_runs(self):
        """Status command runs without crashing."""
        result = runner.invoke(app, ["services", "status"])
        assert result.exit_code == 0


class TestServicesHealth:
    def test_missing_endpoint(self):
        """Health check for nonexistent endpoint should fail."""
        result = runner.invoke(app, ["services", "health", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestServicesDeploy:
    def test_missing_endpoint(self):
        """Deploy to nonexistent endpoint should fail."""
        result = runner.invoke(app, ["services", "deploy", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestServicesStop:
    def test_missing_endpoint(self):
        """Stop on nonexistent endpoint should fail."""
        result = runner.invoke(app, ["services", "stop", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output
