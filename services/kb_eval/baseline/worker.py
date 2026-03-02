"""Subprocess worker for kbEval kernel evaluation.

Spawned by the server for each evaluation request. Compiles, verifies correctness,
and benchmarks kernels in an isolated process.

Usage:
    python -m services.kb_eval.baseline.worker \
        --wd <dir> --reference <file> --generated <file> \
        --device cuda:0 --code-type triton --max-critical-time 120
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import sys
from pathlib import Path

from services.models import KernelExecResult

logger = logging.getLogger(__name__)


def _acquire_device_lock(device: str, lock_dir: Path) -> object:
    """Acquire a per-device file lock to serialize GPU access.

    Returns the lock file object (must be kept alive for lock duration).
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe_name = device.replace(":", "_").replace("/", "_")
    lock_path = lock_dir / f".lock_{safe_name}"
    lock_file = open(lock_path, "w")  # noqa: SIM115
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    return lock_file


def _release_device_lock(lock_file: object) -> None:
    """Release the device lock."""
    try:
        fcntl.flock(lock_file, fcntl.LOCK_UN)  # type: ignore[arg-type]
        lock_file.close()  # type: ignore[union-attr]
    except Exception:
        pass


def run_eval(
    working_dir: Path,
    reference_file: Path,
    generated_file: Path,
    device: str = "cuda:0",
    code_type: str = "triton",
    max_critical_time: int = 120,
    eval_ref_only: bool = False,
) -> KernelExecResult:
    """Run the full evaluation pipeline.

    Args:
        working_dir: Directory for intermediate files.
        reference_file: Path to reference code file.
        generated_file: Path to generated code file.
        device: CUDA device string.
        code_type: One of "triton", "cuda", "pytorch".
        max_critical_time: Max seconds for GPU-critical section.
        eval_ref_only: If True, only benchmark reference code.

    Returns:
        KernelExecResult with compilation, correctness, and timing data.
    """
    import signal

    from services.kb_eval.baseline.util import (
        graceful_cleanup,
        load_custom_model,
        load_model_and_inputs,
        resolve_triton_code,
        time_execution,
        verify_correctness,
    )

    result = KernelExecResult()
    context: dict = {}
    lock_file = None

    try:
        # 1. Load reference model (import as module for Triton JIT support)
        context = {"__builtins__": __builtins__}

        try:
            import torch

            context["torch"] = torch
        except ImportError:
            pass

        model_cls, get_inputs, get_init_inputs = load_model_and_inputs(
            reference_file, context
        )

        if eval_ref_only:
            # Just benchmark reference
            lock_file = _acquire_device_lock(
                device, Path("~/.reflection").expanduser()
            )

            # Set timeout for GPU section
            def _timeout_handler(signum, frame):
                raise TimeoutError("GPU critical section timed out")

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max_critical_time)

            try:
                init_inputs = get_init_inputs()
                init_inputs = [
                    x.to(device) if hasattr(x, "to") else x for x in init_inputs
                ]
                ref_model = model_cls(*init_inputs).to(device)
                result.compiled = True

                inputs = get_inputs()
                inputs = [x.to(device) if hasattr(x, "to") else x for x in inputs]
                stats = time_execution(ref_model, inputs, device)
                result.runtime = stats["mean_ms"]
                result.runtime_stats = stats
                result.correctness = True
            finally:
                signal.alarm(0)
            return result

        # 2. Load generated model (import as module for Triton JIT support)
        new_model_cls = load_custom_model(generated_file, context, code_type)

        # 3. Validate code structure
        if code_type == "triton":
            gen_code = generated_file.read_text()
            validation = resolve_triton_code(gen_code)
            result.metadata["validation"] = validation

        result.compiled = True

        # 4. Acquire device lock
        lock_file = _acquire_device_lock(
            device, Path("~/.reflection").expanduser()
        )

        # Set timeout for GPU section
        def _timeout_handler(signum, frame):
            raise TimeoutError("GPU critical section timed out")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(max_critical_time)

        try:
            # 5. Instantiate models on GPU
            init_inputs = get_init_inputs()
            init_inputs = [
                x.to(device) if hasattr(x, "to") else x for x in init_inputs
            ]
            ref_model = model_cls(*init_inputs).to(device)
            new_model = new_model_cls(*init_inputs).to(device)

            # 5b. Copy weights from reference to generated model
            # This ensures both models use the same learned parameters,
            # so we're comparing operations, not random initializations.
            import torch

            ref_sd = ref_model.state_dict()
            new_sd = new_model.state_dict()
            # Only copy keys that exist in both and have matching shapes
            copied = {}
            for key in ref_sd:
                if key in new_sd and ref_sd[key].shape == new_sd[key].shape:
                    copied[key] = ref_sd[key]
            if copied:
                new_model.load_state_dict(copied, strict=False)
                logger.info(
                    "Copied %d/%d weight tensors from reference to generated model",
                    len(copied),
                    len(ref_sd),
                )

            # 6. Correctness verification
            corr = verify_correctness(ref_model, new_model, get_inputs, device)
            result.correctness = corr.passed_trials == corr.total_trials
            result.metadata["correctness_detail"] = corr.model_dump()

            # 7. Performance benchmarks (only if correct)
            if result.correctness:
                inputs = get_inputs()
                inputs = [x.to(device) if hasattr(x, "to") else x for x in inputs]

                new_stats = time_execution(new_model, inputs, device)
                ref_stats = time_execution(ref_model, inputs, device)

                result.runtime = new_stats["mean_ms"]
                result.runtime_stats = {
                    "generated": new_stats,
                    "reference": ref_stats,
                }
        finally:
            signal.alarm(0)

    except Exception as e:
        logger.exception("Evaluation failed")
        result.metadata["error"] = str(e)
        result.metadata["error_type"] = type(e).__name__
    finally:
        # 8. Cleanup
        if lock_file is not None:
            _release_device_lock(lock_file)
        graceful_cleanup(context, device)

    return result


def main() -> None:
    """CLI entry point for the worker subprocess."""
    parser = argparse.ArgumentParser(description="kbEval subprocess worker")
    parser.add_argument("--wd", type=Path, required=True, help="Working directory")
    parser.add_argument("--reference", type=Path, required=True, help="Reference code file")
    parser.add_argument("--generated", type=Path, default=None, help="Generated code file")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--code-type", type=str, default="triton", help="Code type")
    parser.add_argument("--max-critical-time", type=int, default=120, help="GPU lock timeout")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")
    parser.add_argument("--eval-ref-only", action="store_true", help="Only benchmark reference")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    eval_ref_only = args.eval_ref_only or args.generated is None

    result = run_eval(
        working_dir=args.wd,
        reference_file=args.reference,
        generated_file=args.generated if not eval_ref_only else args.reference,
        device=args.device,
        code_type=args.code_type,
        max_critical_time=args.max_critical_time,
        eval_ref_only=eval_ref_only,
    )

    output_path = args.output or (args.wd / "result.json")
    output_path.write_text(json.dumps(result.model_dump(), indent=2))

    # Also print to stdout for the server to capture
    print(json.dumps(result.model_dump()))


if __name__ == "__main__":
    main()
    sys.exit(0)
