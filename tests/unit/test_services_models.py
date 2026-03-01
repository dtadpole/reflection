"""Tests for services data models."""

from services.models import (
    CorrectnessResult,
    KernelExecResult,
    ServiceHealth,
    ServiceStatus,
)


class TestKernelExecResult:
    def test_defaults(self):
        r = KernelExecResult()
        assert r.compiled is False
        assert r.correctness is False
        assert r.runtime == -1.0
        assert r.metadata == {}
        assert r.runtime_stats == {}

    def test_successful_result(self):
        r = KernelExecResult(
            compiled=True,
            correctness=True,
            runtime=1.23,
            metadata={"error": None},
            runtime_stats={"mean_ms": 1.23, "min_ms": 1.0},
        )
        assert r.compiled is True
        assert r.correctness is True
        assert r.runtime == 1.23

    def test_roundtrip(self):
        r = KernelExecResult(compiled=True, runtime=5.0)
        data = r.model_dump()
        r2 = KernelExecResult.model_validate(data)
        assert r2.compiled is True
        assert r2.runtime == 5.0


class TestCorrectnessResult:
    def test_defaults(self):
        c = CorrectnessResult()
        assert c.total_trials == 0
        assert c.passed_trials == 0
        assert c.max_diff == -1.0
        assert c.avg_diff == -1.0

    def test_all_passed(self):
        c = CorrectnessResult(total_trials=3, passed_trials=3, max_diff=0.001, avg_diff=0.0005)
        assert c.passed_trials == c.total_trials

    def test_partial_pass(self):
        c = CorrectnessResult(total_trials=3, passed_trials=1, max_diff=0.1, avg_diff=0.05)
        assert c.passed_trials < c.total_trials


class TestServiceHealth:
    def test_defaults(self):
        h = ServiceHealth(name="test")
        assert h.name == "test"
        assert h.status == ServiceStatus.UNKNOWN
        assert h.devices == []
        assert h.pending_requests == 0

    def test_running(self):
        h = ServiceHealth(
            name="gpu-1",
            status=ServiceStatus.RUNNING,
            endpoint="http://gpu1:8456",
            devices=["cuda:0", "cuda:1"],
            pending_requests=2,
        )
        assert h.status == ServiceStatus.RUNNING
        assert len(h.devices) == 2


class TestServiceStatus:
    def test_values(self):
        assert ServiceStatus.RUNNING == "running"
        assert ServiceStatus.STOPPED == "stopped"
        assert ServiceStatus.ERROR == "error"
        assert ServiceStatus.UNKNOWN == "unknown"
