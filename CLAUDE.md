# Reflection

Self-evolving autonomous multi-agent system that accumulates a knowledge base and improves its ability to solve coding problems.

## Build & Run

```bash
uv sync                                    # Install dependencies
uv run reflection run --iterations 3       # Run autonomous loop
uv run reflection solve "problem desc"     # Solve a single problem
uv run reflection status                   # Show system status
uv run reflection cards list               # List knowledge cards
uv run reflection cards search "query"     # Search cards
uv run reflection trajectories list        # List solver trajectories
```

## Test & Lint

```bash
uv run pytest tests/unit/                  # Unit tests
uv run pytest tests/integration/           # Integration tests
uv run pytest                              # All tests
uv run ruff check .                        # Lint
uv run ruff check . --fix                  # Auto-fix lint issues
```

## Project Structure

- `agenix/` — Agent execution framework (Claude Agent SDK runtime, orchestration, tools, storage)
- `agents/` — Agent definitions (markdown + YAML + Python), each agent is a subfolder
- `cli/` — Typer CLI entry point
- `config/` — Hydra YAML configs (system-level)
- `DESIGN.md` — Architecture diagrams and pipeline flow

## Configuration

- All config via **Hydra** (`hydra-core` + `omegaconf`), YAML only
- System config: `config/config.yaml`
- Agent configs: `agents/<name>/config.yaml` (Hydra config groups)

## Agent Definitions

Each agent lives in `agents/<name>/` with:

- **`agent.md`** (required) — Sections: `# Name`, `## Description`, `## System Prompt`, `## Input Format`, `## Output Format`, `## Examples`
- **`config.yaml`** (required) — `model`, `temperature`, `max_turns`, `tools`, `custom_tools`
- **`logic.py`** (optional) — Hardened Python logic (parsing, validation)
- **`tools.md`** (optional) — Detailed tool specifications

Loader (`agenix/loader.py`) parses `agent.md` sections, reads `config.yaml`, detects `logic.py`.

## Data Storage

Path: `<reflection_data_root>/<reflection_env>/<run_tag>/<agent_name>/`

- Default `reflection_data_root`: `~/.reflection`
- Available `reflection_env`: `prod`, `int`, `test_${USER}`
- Default `run_tag`: `run_{YYYYMMDD_HHMMSS}`
- SQLite and ChromaDB are **per-env shared** across runs
- Agent outputs are **per-run** under `<run_tag>/<agent_name>/`

## Key Patterns

- Custom tools: `@tool` decorator + `create_sdk_mcp_server` from `claude-agent-sdk`
- Tool references: `mcp__<server-key>__<tool-name>` in `allowed_tools`
- Data models: Pydantic v2 in `agenix/storage/models.py`
- Storage: SQLite (relational) + ChromaDB (vector), both embedded
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`), local CPU
- Pipeline: CURATOR → SOLVER → REFLECTOR → ORGANIZER (+ INSIGHT_FINDER periodically)
