# Verifier Tool

The verifier is a tool that evaluates generated code against a reference implementation. Different variants connect to different backends.

## Design Pattern

The verifier follows a **variant-as-backend** pattern: the tool name (`verifier`) stays constant across agents, but the variant determines which evaluation backend is used. This allows agent configs to simply reference `verifier` while the bootstrap layer selects the appropriate variant based on available infrastructure.

## Variants

### `kb_eval` (current)
- **Backend**: Remote kbEval FastAPI service running on GPU hosts
- **Use case**: GPU kernel verification (Triton/CUDA/PyTorch)
- **Dependencies**: `KbEvalClient` (requires configured service endpoint)
- **MCP tool name**: `verifier`

### Future variants (planned)
- `local` — Local subprocess execution (CPU-only, no GPU required)
- `docker` — Containerized evaluation with GPU passthrough

## How It Works

1. Agent config lists `verifier` in `custom_tools`
2. Bootstrap (`cli/main.py`) calls `load_tool("verifier", variant="kb_eval")`
3. The loader imports `logic.py` and extracts `create_tool()`
4. Bootstrap calls `create_tool(kb_eval_client=client)` to get an `SdkMcpTool`
5. The tool is registered in `ToolRegistry` under name `verifier`
6. Agent accesses it as `mcp__reflection__verifier`

## Adding a New Variant

1. Create `tools/verifier/<variant>/` with `tool.md`, `config.yaml`, `logic.py`
2. `logic.py` must export `create_tool(**kwargs) -> SdkMcpTool`
3. The `@tool()` decorator name should be `verifier` (consistent across variants)
4. Update bootstrap to select variant based on config/environment
