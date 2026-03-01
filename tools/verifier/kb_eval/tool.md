# Verifier (kb_eval)

## Description
Evaluate a generated GPU kernel against a reference implementation by sending
both to a remote kbEval server. Returns compilation status, correctness
(torch.allclose), and runtime benchmarks.

## Input Schema
- reference_code (str, required): Reference Python code with Model class
- generated_code (str, required): Generated code with ModelNew class
- code_type (str, optional): "triton" | "cuda" | "pytorch" (default: "triton")

## Output Schema
- compiled (bool): Whether the generated code compiled
- correctness (bool): Whether outputs match reference (atol=1e-2)
- runtime (float): Execution time in milliseconds (-1 if failed)
- metadata (dict): Validation details, correctness breakdown
- runtime_stats (dict): Timing statistics for both generated and reference

## Examples
Input:
```json
{
  "reference_code": "class Model(torch.nn.Module): ...",
  "generated_code": "class ModelNew(torch.nn.Module): ...",
  "code_type": "triton"
}
```

Output:
```json
{
  "compiled": true,
  "correctness": true,
  "runtime": 0.42,
  "metadata": {"max_diff": 0.001, "allclose": true},
  "runtime_stats": {"generated_ms": 0.42, "reference_ms": 0.85}
}
```
