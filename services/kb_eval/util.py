"""Core evaluation logic for kbEval kernel verification.

Handles loading reference/generated models, AST validation of Triton code,
correctness verification, and GPU timing.
"""

from __future__ import annotations

import ast
import gc
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from services.models import CorrectnessResult

logger = logging.getLogger(__name__)


def _import_module_from_file(file_path: Path, module_name: str) -> ModuleType:
    """Import a Python file as a module using importlib.

    This is required for Triton @jit functions, which need file-backed source.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot create module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model_and_inputs(
    reference_code: str | Path,
    context: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Load reference code and extract Model, get_inputs, get_init_inputs.

    When given a Path, imports the file as a module (required for Triton JIT).
    When given a string, uses exec() (for tests / non-GPU contexts).

    Args:
        reference_code: Path to reference .py file, or source string.
        context: Exec namespace (updated with loaded symbols).

    Returns:
        Tuple of (Model class, get_inputs function, get_init_inputs function).

    Raises:
        ValueError: If required symbols are not found.
    """
    if isinstance(reference_code, Path):
        mod = _import_module_from_file(reference_code, "_kb_eval_reference")
        context["_ref_module"] = mod
        model_cls = getattr(mod, "Model", None)
        get_inputs = getattr(mod, "get_inputs", None)
        get_init_inputs = getattr(mod, "get_init_inputs", None)
        # Also put them in context for downstream use
        if model_cls:
            context["Model"] = model_cls
        if get_inputs:
            context["get_inputs"] = get_inputs
        if get_init_inputs:
            context["get_init_inputs"] = get_init_inputs
    else:
        compiled = compile(reference_code, "<reference>", "exec")
        exec(compiled, context)  # noqa: S102
        model_cls = context.get("Model")
        get_inputs = context.get("get_inputs")
        get_init_inputs = context.get("get_init_inputs")

    if model_cls is None:
        raise ValueError("Reference code must define a 'Model' class")
    if get_inputs is None:
        raise ValueError("Reference code must define a 'get_inputs' function")
    if get_init_inputs is None:
        raise ValueError("Reference code must define a 'get_init_inputs' function")

    return model_cls, get_inputs, get_init_inputs


def load_custom_model(
    generated_code: str | Path,
    context: dict[str, Any],
    code_type: str = "triton",
) -> Any:
    """Load generated ModelNew, validating it doesn't clobber originals.

    When given a Path, imports the file as a module (required for Triton JIT).
    When given a string, uses exec() (for tests / non-GPU contexts).

    Args:
        generated_code: Path to generated .py file, or source string.
        context: Exec namespace (should already contain reference Model).
        code_type: One of "triton", "cuda", "pytorch".

    Returns:
        ModelNew class.

    Raises:
        ValueError: If ModelNew not defined or original Model was modified.
    """
    original_model = context.get("Model")

    if isinstance(generated_code, Path):
        mod = _import_module_from_file(generated_code, "_kb_eval_generated")
        context["_gen_module"] = mod
        new_model_cls = getattr(mod, "ModelNew", None)
        # Check if generated module redefined Model
        gen_model = getattr(mod, "Model", None)
        if gen_model is not None and gen_model is not original_model:
            raise ValueError(
                "Generated code must not modify the original 'Model' class"
            )
    else:
        compiled = compile(generated_code, "<generated>", "exec")
        exec(compiled, context)  # noqa: S102
        new_model_cls = context.get("ModelNew")
        if context.get("Model") is not original_model:
            raise ValueError(
                "Generated code must not modify the original 'Model' class"
            )

    if new_model_cls is None:
        raise ValueError("Generated code must define a 'ModelNew' class")

    return new_model_cls


def resolve_triton_code(code: str) -> dict[str, Any]:
    """AST-based validation of Triton kernel code.

    Checks for:
    - At least one @triton.jit decorated function
    - A ModelNew class definition
    - ModelNew.forward() method that calls a jit-decorated function

    Returns:
        Dict with 'jit_functions' (list of names), 'model_new_class' (name),
        'forward_calls' (jit functions called from forward).

    Raises:
        ValueError: If validation fails.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {e}") from e

    jit_functions: list[str] = []
    model_new_class: str | None = None
    forward_calls: list[str] = []

    for node in ast.walk(tree):
        # Find @triton.jit decorated functions
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if _is_triton_jit(dec):
                    jit_functions.append(node.name)

        # Find ModelNew class
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            model_new_class = node.name
            # Find forward method and its calls to jit functions
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    for child in ast.walk(item):
                        if isinstance(child, ast.Call):
                            call_name = _get_call_name(child)
                            if call_name:
                                forward_calls.append(call_name)

    if not jit_functions:
        raise ValueError("No @triton.jit decorated functions found")
    if model_new_class is None:
        raise ValueError("No 'ModelNew' class found")

    # Check that forward calls at least one jit function
    jit_called = [f for f in forward_calls if f in jit_functions]
    if not jit_called:
        raise ValueError(
            f"ModelNew.forward() does not call any jit functions. "
            f"jit functions: {jit_functions}, forward calls: {forward_calls}"
        )

    return {
        "jit_functions": jit_functions,
        "model_new_class": model_new_class,
        "forward_calls": forward_calls,
    }


def verify_correctness(
    ref_model: Any,
    new_model: Any,
    get_inputs_fn: Any,
    device: str,
    num_trials: int = 3,
) -> CorrectnessResult:
    """Run both models and compare outputs with torch.allclose.

    Args:
        ref_model: Instantiated reference model on device.
        new_model: Instantiated generated model on device.
        get_inputs_fn: Function returning input tensors.
        device: CUDA device string (e.g. "cuda:0").
        num_trials: Number of comparison trials.

    Returns:
        CorrectnessResult with trial counts and diff stats.
    """
    import torch

    passed = 0
    max_diff = 0.0
    total_diff = 0.0

    for _ in range(num_trials):
        inputs = get_inputs_fn()
        inputs = [x.to(device) if hasattr(x, "to") else x for x in inputs]

        with torch.no_grad():
            ref_out = ref_model(*inputs)
            new_out = new_model(*inputs)

        if isinstance(ref_out, torch.Tensor):
            ref_out = [ref_out]
            new_out = [new_out]

        trial_passed = True
        for r, n in zip(ref_out, new_out):
            if not isinstance(r, torch.Tensor):
                continue
            diff = (r.float() - n.float()).abs()
            trial_max = diff.max().item()
            trial_avg = diff.mean().item()
            max_diff = max(max_diff, trial_max)
            total_diff += trial_avg

            if not torch.allclose(r.float(), n.float(), atol=1e-2, rtol=1e-2):
                trial_passed = False

        if trial_passed:
            passed += 1

    return CorrectnessResult(
        total_trials=num_trials,
        passed_trials=passed,
        max_diff=max_diff,
        avg_diff=total_diff / num_trials if num_trials > 0 else 0.0,
    )


def time_execution(
    model: Any,
    inputs: list[Any],
    device: str,
    num_warmups: int = 5,
    num_trials: int = 10,
) -> dict[str, float]:
    """Time kernel execution using CUDA events.

    Args:
        model: Instantiated model on device.
        inputs: Input tensors (already on device).
        device: CUDA device string.
        num_warmups: Warmup iterations.
        num_trials: Timed iterations.

    Returns:
        Dict with 'mean_ms', 'min_ms', 'max_ms', 'std_ms' keys.
    """
    import torch

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmups):
            model(*inputs)

    torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(num_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            model(*inputs)
        end.record()

        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    std_ms = variance**0.5

    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
    }


def graceful_cleanup(context: dict[str, Any], device: str) -> None:
    """Delete execution context and free CUDA memory.

    Args:
        context: The exec namespace to clean up.
        device: CUDA device string.
    """
    context.clear()
    gc.collect()

    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _is_triton_jit(decorator: ast.expr) -> bool:
    """Check if a decorator is @triton.jit."""
    if isinstance(decorator, ast.Attribute):
        if decorator.attr == "jit" and isinstance(decorator.value, ast.Name):
            return decorator.value.id == "triton"
    if isinstance(decorator, ast.Name):
        return decorator.id == "triton_jit"
    return False


def _get_call_name(node: ast.Call) -> str | None:
    """Extract function name from an AST Call node.

    Handles:
    - Direct calls: foo(...)  -> "foo"
    - Attribute calls: self.foo(...)  -> "foo"
    - Subscript calls: foo[grid](...)  -> "foo"  (Triton kernel launch pattern)
    """
    func = node.func
    # Triton kernel launch: kernel[grid](...) — func is a Subscript
    if isinstance(func, ast.Subscript):
        func = func.value
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None
