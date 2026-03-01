"""FastAPI server for kbEval kernel evaluation.

Accepts evaluation requests, spawns subprocess workers, and returns results.

Run with: uvicorn services.kb_eval.baseline.server:app --host 0.0.0.0 --port 8456
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from services.models import KernelExecResult, ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)

app = FastAPI(title="kbEval", version="1.0.0")

# Server state
_devices: list[str] = ["cuda:0"]
_pending: dict[str, int] = {}  # device -> count
_max_critical_time: int = 120
_max_timeout: int = 600
_code_type: str = "triton"
_port: int = 8456
_data_root: Path = Path("~/.reflection").expanduser() / "kb_eval"


# --- Request/Response models ---


class EvalRequest(BaseModel):
    reference_code: str
    generated_code: str
    run_tag: str = ""
    task_tag: str = ""
    code_type: str = "triton"
    device: Optional[str] = None


class EvalRefRequest(BaseModel):
    reference_code: str
    run_tag: str = ""
    task_tag: str = ""
    device: Optional[str] = None


# --- Helpers ---


def _select_device(preferred: str | None) -> str:
    """Select the device with the fewest pending requests."""
    if preferred and preferred in _devices:
        return preferred
    if not _devices:
        return "cuda:0"
    return min(_devices, key=lambda d: _pending.get(d, 0))


def _make_work_dir(run_tag: str, task_tag: str) -> Path:
    """Create a working directory for this evaluation."""
    if not run_tag:
        run_tag = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    if not task_tag:
        task_tag = uuid.uuid4().hex[:12]
    wd = _data_root / run_tag / task_tag
    wd.mkdir(parents=True, exist_ok=True)
    return wd


async def _run_worker(
    wd: Path,
    reference_file: Path,
    generated_file: Path | None,
    device: str,
    code_type: str,
    eval_ref_only: bool = False,
) -> KernelExecResult:
    """Spawn the worker subprocess and collect its result."""
    cmd = [
        sys.executable, "-m", "services.kb_eval.baseline.worker",
        "--wd", str(wd),
        "--reference", str(reference_file),
        "--device", device,
        "--code-type", code_type,
        "--max-critical-time", str(_max_critical_time),
        "--output", str(wd / "result.json"),
    ]

    if eval_ref_only:
        cmd.append("--eval-ref-only")
    elif generated_file is not None:
        cmd.extend(["--generated", str(generated_file)])

    _pending[device] = _pending.get(device, 0) + 1

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_max_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return KernelExecResult(
                metadata={"error": "Worker timed out", "timeout": _max_timeout}
            )

        if proc.returncode != 0:
            return KernelExecResult(
                metadata={
                    "error": "Worker exited with non-zero status",
                    "exit_code": proc.returncode,
                    "stderr": stderr.decode(errors="replace")[-2000:],
                }
            )

        # Read result from output file
        result_path = wd / "result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text())
            return KernelExecResult.model_validate(data)

        # Fallback: parse stdout
        out = stdout.decode(errors="replace").strip()
        if out:
            last_line = out.splitlines()[-1]
            data = json.loads(last_line)
            return KernelExecResult.model_validate(data)

        return KernelExecResult(metadata={"error": "No result produced"})

    finally:
        _pending[device] = max(0, _pending.get(device, 1) - 1)


# --- Endpoints ---


@app.post("/eval", response_model=KernelExecResult)
async def eval_kernel(req: EvalRequest) -> KernelExecResult:
    """Evaluate generated kernel code against a reference implementation."""
    device = _select_device(req.device)
    wd = _make_work_dir(req.run_tag, req.task_tag)

    ref_file = wd / "reference.py"
    gen_file = wd / "generated.py"
    ref_file.write_text(req.reference_code)
    gen_file.write_text(req.generated_code)

    return await _run_worker(
        wd=wd,
        reference_file=ref_file,
        generated_file=gen_file,
        device=device,
        code_type=req.code_type,
    )


@app.post("/eval_ref", response_model=KernelExecResult)
async def eval_reference(req: EvalRefRequest) -> KernelExecResult:
    """Benchmark reference code only (no generated code comparison)."""
    device = _select_device(req.device)
    wd = _make_work_dir(req.run_tag, req.task_tag)

    ref_file = wd / "reference.py"
    ref_file.write_text(req.reference_code)

    return await _run_worker(
        wd=wd,
        reference_file=ref_file,
        generated_file=None,
        device=device,
        code_type="pytorch",
        eval_ref_only=True,
    )


@app.get("/health", response_model=ServiceHealth)
async def health() -> ServiceHealth:
    """Health check endpoint."""
    return ServiceHealth(
        name="kb_eval",
        status=ServiceStatus.RUNNING,
        endpoint=f"http://0.0.0.0:{_get_port()}",
        devices=list(_devices),
        pending_requests=sum(_pending.values()),
    )


@app.get("/stats")
async def stats() -> dict:
    """Server statistics."""
    return {
        "devices": list(_devices),
        "device_count": len(_devices),
        "pending_per_device": dict(_pending),
        "total_pending": sum(_pending.values()),
        "code_type": _code_type,
    }


def _get_port() -> int:
    """Return the configured port."""
    return _port


def configure(
    devices: list[str] | None = None,
    max_critical_time: int = 120,
    max_timeout: int = 600,
    code_type: str = "triton",
    data_root: str | None = None,
    port: int = 8456,
) -> None:
    """Configure server settings (call before starting uvicorn)."""
    global _devices, _max_critical_time, _max_timeout, _code_type, _data_root, _port

    if devices:
        _devices = devices
    _max_critical_time = max_critical_time
    _max_timeout = max_timeout
    _code_type = code_type
    _port = port
    if data_root:
        _data_root = Path(data_root).expanduser() / "kb_eval"

    # Initialize pending counters
    for d in _devices:
        _pending.setdefault(d, 0)
