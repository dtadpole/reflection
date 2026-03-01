# Curator

## Description
Pure Python KernelBench loader — no LLM required.

## System Prompt
This agent has no system prompt. The curator is implemented as pure Python that samples problems from the KernelBench dataset (ScalingIntelligence/KernelBench on HuggingFace).

## How It Works

The curator is not an LLM agent. It is a Python function (`agenix/agents/curator_handler.py`) that:

1. Loads the KernelBench dataset from HuggingFace (270 PyTorch GPU kernel problems)
2. Randomly samples N problems from specified difficulty levels
3. Converts each to a `Problem` model with `reference_code` (full PyTorch source)
4. Saves to FSBackend and enqueues to the `problems` queue
5. Deduplicates by title to handle restarts

### KernelBench Dataset

- **Source**: `ScalingIntelligence/KernelBench` on HuggingFace
- **Size**: 270 problems across 4 levels
  - `level_1`: 100 problems (easy — single ops like ReLU, softmax)
  - `level_2`: 100 problems (medium — fused ops, small models)
  - `level_3`: 50 problems (hard — multi-op fusion, complex patterns)
  - `level_4`: 20 problems (hard — full model conversions)
- **Each row**: `{code, level, name, problem_id}`
  - `code` is a complete PyTorch module with `Model(nn.Module)`, `get_inputs()`, `get_init_inputs()`

### CLI Usage

```bash
reflection agent curator -n 100 --levels level_1,level_2 --seed 42 -v
```

## Input Format
N/A (pure Python, no LLM input)

## Output Format
N/A (pure Python, no LLM output)
