# agenix — Agent Execution Framework

Generic framework that uses `claude-agent-sdk` to execute agents defined in `agents/`.

## How the Loader Works

`loader.py` reads agent folders:
- `agent.md` → parsed into system prompt, description, I/O format
- `config.yaml` → model, temperature, max_turns, tools, custom_tools
- `logic.py` → optional Python module with hardened logic (imported dynamically)

## Custom Tools (MCP Server Pattern)

Tools are registered in `tools/registry.py`. Each tool uses the `@tool` decorator from `claude-agent-sdk`:

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("tool_name", "description", {"param": str})
async def my_tool(args: dict) -> dict:
    return {"content": [{"type": "text", "text": "result"}]}

server = create_sdk_mcp_server(name="server", tools=[my_tool])
```

Tools are referenced as `mcp__<server-key>__<tool-name>` in `allowed_tools`.

## Storage

- `storage/models.py` — Pydantic data models (Problem, Trajectory, Card, etc.)
- `storage/fs_backend.py` — Filesystem CRUD (JSON files) + DuckDB query engine

Knowledge store, embedder, and vector index live in `tools/knowledge/baseline/`.

## Pipeline

`pipeline.py` orchestrates agent sequence: Curator → Solver → Critic → Organizer.
`scheduler.py` controls InsightFinder frequency.
