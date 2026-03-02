# Reflection

Self-evolving autonomous multi-agent system for GPU kernel optimization. Agents operate independently, communicate via filesystem queues and a shared knowledge base, and accumulate Triton kernel optimization knowledge over time.

## Build & Run

```bash
uv sync                                    # Install dependencies
uv run reflection status                   # Show system status
uv run reflection cards list               # List knowledge cards
uv run reflection cards search "query"     # Search cards
uv run reflection experiences list          # List solver experiences
```

### Running Agents (async, each in its own terminal)

```bash
# 1. Load problems from KernelBench
uv run reflection agent curator -n 100 --levels level_1,level_2 -v

# 2. Check queue status
uv run reflection queues status

# 3. Start solver (requires SSH tunnels for verifier + embedder)
reflection services tunnel start
uv run reflection agent solver -v                # Single solver
uv run reflection agent solver --parallel 3 -v   # Parallel (3 instances per problem)

# 4. Start critic (in another terminal)
uv run reflection agent critic -v

# 5. Start organizer (in another terminal)
uv run reflection agent organizer --interval 300 -v

# 6. Start insight finder (in another terminal)
uv run reflection agent insight-finder --interval 600 -v
```

### Legacy Sequential Pipeline

```bash
uv run reflection run --iterations 3       # Run sequential pipeline
uv run reflection solve "problem desc"     # Solve a single problem
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
  - `agenix/agents/` — Agent handlers (curator, solver, parallel_solver, critic, organizer, insight finder)
  - `agenix/agent_loop.py` — QueueAgentLoop + ScheduledAgentLoop abstractions
  - `agenix/runner.py` — ClaudeRunner (claude-agent-sdk query), supports `log_name` for parallel labeling
  - `agenix/parsers.py` — Output parsers for agent responses
  - `agenix/queue/` — Filesystem-based message queues (FSQueue)
  - `agenix/storage/lineage.py` — Card lineage operations (create/revise/merge/split/archive)
- `agents/` — Agent definitions (markdown + YAML + Python), each agent is a subfolder
  - `agents/critic/base/` — Single-experience critic
  - `agents/critic/batch/` — Batch comparative critic (for parallel solver)
- `tools/` — Tool definitions (tool.md + config.yaml + logic.py), each tool is a subfolder
- `cli/` — Typer CLI entry point
- `config/` — Hydra YAML configs (system-level)
- `DESIGN.md` — Architecture diagrams, card lifecycle, and pipeline flow

## Architecture: Async Queue-Based

Agents operate independently and communicate via filesystem queues and the shared knowledge base.

```
                  CURATOR
                     │
                     ▼
                 [problems]
                     │
                     ▼
  <Verifier> ◀──── SOLVER ◀────────── <Retriever>
     │             ▲ │                     ▲
     └─────────────┘ │                     │
       (iterate)     ▼                     │
                 [experiences]             │
                       │                   │
                       ▼                   │
                    CRITIC                 │
                    │    │                 │
                    │    ▼                 │
                    │   [Knowledge Base] ──┘
                    │       ▲        ▲
                    ▼       │        │
            [reflections]   │        │
                    │       │        │
                    ▼       │        │
                    ORGANIZER   INSIGHT_FINDER
                    (periodic)   (periodic)
```

### Queue Topology

Three queues (`problems`, `experiences`, `reflections`) under `<data_root>/<env>/queues/`:

| Queue | Producer | Consumer | Payload |
|-------|----------|----------|---------|
| `problems` | CURATOR | SOLVER | `{problem_id, title}` |
| `experiences` | SOLVER | CRITIC | `{problem_id, experience_ids: [...]}` (parallel) or `{experience_id, problem_id}` (single) |
| `reflections` | CRITIC | (future) | `{card_id}` |

Each queue has subdirectories: `pending/`, `processing/`, `done/`, `failed/`.
Messages are JSON files. State transitions use atomic `os.rename()` for POSIX safety.

### Agent Roles

| Agent | Type | Description |
|-------|------|-------------|
| CURATOR | One-shot (pure Python) | Loads KernelBench problems from HuggingFace, enqueues to `problems` |
| SOLVER | Queue loop | Dequeues problems, writes Triton kernels, verifies, enqueues experiences. Supports `--parallel N` for concurrent instances per problem. |
| CRITIC | Queue loop | Dequeues experiences, produces reflection cards via `knowledge_create` tool. Auto-selects batch variant for multi-experience payloads. |
| ORGANIZER | Scheduled (5 min) | Synthesizes knowledge cards from recent experiences + reflections |
| INSIGHT_FINDER | Scheduled (10 min) | Detects cross-cutting meta-patterns across experiences |

### Problem Source: KernelBench

Problems come from [ScalingIntelligence/KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) — 270 PyTorch GPU kernel problems:
- `level_1`: 100 easy (single ops)
- `level_2`: 100 medium (fused ops)
- `level_3`: 50 hard (multi-op fusion)
- `level_4`: 20 hard (full model conversion)

Each problem contains reference PyTorch code. The solver writes Triton kernel replacements.

### Agent Loop Abstractions

- **`QueueAgentLoop`**: Polls FSQueue, dispatches to handler, auto-completes/fails messages. Exponential backoff on empty queue. Graceful SIGINT/SIGTERM shutdown.
- **`ScheduledAgentLoop`**: Runs handler on a timer interval. Used by ORGANIZER and INSIGHT_FINDER.

### SOLVER Retrieval Strategy

The solver retrieves 7-10 knowledge cards using its full working context as the retrieval query:
- Problem title + description + domain
- Previous attempts and feedback
- Current plan of attack
- Query is passed to the retriever (with rerank via Qwen3-32B cross-encoder)

## Configuration

- System config: `config/default.toml` + `config/hosts.yaml` + `config/tunnels.yaml`
- Agent configs: `agents/<name>/<variant>/config.yaml`

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
- **Shared data** (`problems/`, `cards/`, `experiences/`, `lance/`, `queues/`) at env level, persists across runs
- **DuckDB** as query engine over JSON files (no persistent DB)
- **LanceDB** for vector embeddings (semantic search over cards)

## Design Rules

- **Tool-mediated verification**: The SOLVER must always use the `verifier` tool for correctness and performance checks. It must never attempt its own verification — no manual testing, no SSH to GPU hosts, no writing benchmark scripts, no running code locally. The verifier is the single source of truth.

## Key Patterns

- Custom tools: `@tool` decorator + `create_sdk_mcp_server` from `claude-agent-sdk`
- Tool references: `mcp__<server-key>__<tool-name>` in `allowed_tools`
- Data models: Pydantic v2 in `agenix/storage/models.py`
- Storage: JSON files (filesystem) + DuckDB (query) + LanceDB (vector)
- Knowledge tools: 8 individual MCP tools in `tools/knowledge/baseline/logic.py` (search, list, get, create, revise, merge, split, archive)
- Recall tools: 3 MCP tools in `tools/recall/baseline/logic.py` (fetch, outline, excerpt) for reading experiences/problems
- Embeddings: `sentence-transformers` (local CPU) or remote Qwen3-Embedding-8B (GPU)
- Agents: Independent async processes communicating via filesystem queues + shared knowledge base
- Card lifecycle: CREATE → ACTIVE → REVISE/MERGE/SPLIT → SUPERSEDED, or ARCHIVE → ARCHIVED (see DESIGN.md)
