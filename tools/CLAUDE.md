# Tool Definitions

Tools are organized as `tools/<tool_name>/<variant>/`, mirroring the agent folder pattern.

## Folder Structure

```
tools/<tool_name>/<variant>/
├── tool.md       # Description, input/output schema, examples
├── config.yaml   # Default config (name, variant, tool-specific settings)
└── logic.py      # create_tool(**kwargs) factory function
```

## Required Files

- **tool.md** — Tool definition with sections:
  - `# Tool Name` — title
  - `## Description` — what the tool does
  - `## Input Schema` — parameters the tool accepts
  - `## Output Schema` — what the tool returns
  - `## Examples` — usage examples (optional)

- **config.yaml** — Tool configuration:
  ```toml
  name = "tool_name"
  variant = "base"
  ```

- **logic.py** — Must export a `create_tool(**kwargs)` factory:
  ```python
  def create_tool(*, dependency) -> SdkMcpTool:
      @tool("tool_name", "description", {...})
      async def tool_fn(args: dict) -> dict:
          ...
      return tool_fn
  ```

## Conventions

- `logic.py` must export `create_tool(**kwargs)` as its factory function
- The tool name in the `@tool()` decorator determines the MCP tool name
- Agent configs reference tool names; variant selection happens at bootstrap time
- Runtime dependencies (clients, stores) are injected via `create_tool()` kwargs
- Keep `tool.md` descriptions concise but complete — they serve as documentation
- The `knowledge/baseline/` tool also contains core modules (store.py, embedder.py, index.py) imported by other tools and the pipeline
