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

**Testing rules:**
- Always write **unit tests** when adding new services, tools, or other new components
- Always write **integration tests** for services and tools (test against actual live deployment)

## Project Structure

- `agenix/` — Agent execution framework (Claude Agent SDK runtime, orchestration, tools, storage)
- `agents/` — Agent definitions (markdown + YAML + Python), each agent is a subfolder
- `tools/` — Tool definitions (tool.md + config.yaml + logic.py), each tool is a subfolder
- `cli/` — Typer CLI entry point
- `config/` — Hydra YAML configs (system-level)
- `DESIGN.md` — Architecture diagrams and pipeline flow

## Configuration

- All config via **Hydra** (`hydra-core` + `omegaconf`), YAML only
- System config: `config/config.yaml`
- Agent configs: `agents/<name>/config.yaml` (Hydra config groups)

## Agent Definitions

Agents are organized as `agents/<agent_name>/<variant>/`. The initial variant is `base`.

Each variant folder contains:

- **`agent.md`** (required) — Sections: `# Name`, `## Description`, `## System Prompt`, `## Input Format`, `## Output Format`, `## Examples`
- **`config.yaml`** (required) — `model`, `temperature`, `max_turns`, `tools`, `custom_tools`
- **`logic.py`** (optional) — Hardened Python logic (parsing, validation)
- **`tools.md`** (optional) — Detailed tool specifications

Loader (`agenix/loader.py`) parses `agent.md` sections, reads `config.yaml`, detects `logic.py`.
Default variant is `base` if not specified.

## Data Storage

Path: `<reflection_data_root>/<reflection_env>/`

- Default `reflection_data_root`: `~/.reflection`
- Available `reflection_env`: `prod`, `int`, `test_${USER}`
- Default `run_tag`: `run_{YYYYMMDD_HHMMSS}`
- **Filesystem-based**: all data stored as JSON files, one file per entity
- **Shared data** (`problems/`, `cards/`, `lance/`) at env level, persists across runs
- **Per-run data** under `<run_tag>/<agent_name>/`, one JSON per entity
- **DuckDB** as query engine over JSON files (no persistent DB)
- **LanceDB** for vector embeddings (semantic search over cards)

## Key Patterns

- Custom tools: `@tool` decorator + `create_sdk_mcp_server` from `claude-agent-sdk`
- Tool references: `mcp__<server-key>__<tool-name>` in `allowed_tools`
- Data models: Pydantic v2 in `agenix/storage/models.py`
- Storage: JSON files (filesystem) + DuckDB (query) + LanceDB (vector)
- Knowledge: `tools/knowledge/baseline/` (KnowledgeStore, LanceDB index, embedder)
- Embeddings: `sentence-transformers` (local CPU) or remote Qwen3-Embedding-8B (GPU)
- Pipeline: CURATOR → SOLVER → CRITIC → ORGANIZER (+ INSIGHT_FINDER periodically)
