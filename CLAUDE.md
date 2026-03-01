# Reflection

Self-evolving autonomous multi-agent system for GPU kernel optimization. Agents operate independently, communicate via filesystem queues and a shared knowledge base, and accumulate Triton kernel optimization knowledge over time.

## Build & Run

```bash
uv sync                                    # Install dependencies
uv run reflection status                   # Show system status
uv run reflection cards list               # List knowledge cards
uv run reflection cards search "query"     # Search cards
uv run reflection trajectories list        # List solver trajectories
```

### Running Agents (async, each in its own terminal)

```bash
# 1. Load problems from KernelBench
uv run reflection agent curator -n 100 --levels level_1,level_2 -v

# 2. Check queue status
uv run reflection queues status

# 3. Start solver (requires SSH tunnels for verifier + embedder)
reflection services tunnel start
uv run reflection agent solver -v

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
  - `agenix/agents/` — Agent handlers (curator, solver, critic, organizer, insight finder)
  - `agenix/agent_loop.py` — QueueAgentLoop + ScheduledAgentLoop abstractions
  - `agenix/parsers.py` — Output parsers for agent responses
  - `agenix/queue/` — Filesystem-based message queues (FSQueue)
- `agents/` — Agent definitions (markdown + YAML + Python), each agent is a subfolder
- `tools/` — Tool definitions (tool.md + config.yaml + logic.py), each tool is a subfolder
- `cli/` — Typer CLI entry point
- `config/` — Hydra YAML configs (system-level)
- `DESIGN.md` — Architecture diagrams and pipeline flow

## Architecture: Async Queue-Based

Agents operate independently and communicate via filesystem queues and the shared knowledge base.

```
                    CURATOR (pure Python)
                         │
                         ▼
                  [problems queue]
                         │
                         ▼
<Verifier> ◀──── SOLVER ◀──── <Retriever>
    │             ▲  │              ▲
    └─────────────┘  │              │
       (iterate)     ▼              │
              [trajectories queue]  │
                         │          │
                         ▼          │
                      CRITIC        │
                         │          │
                         ▼          │
                  [Knowledge Base] ─┘
                     ▲        ▲
                     │        │
               ORGANIZER   INSIGHT_FINDER
               (periodic)   (periodic)
```

### Queue Topology

Two queues (`problems`, `trajectories`) under `<data_root>/<env>/queues/`:

| Queue | Producer | Consumer | Payload |
|-------|----------|----------|---------|
| `problems` | CURATOR | SOLVER | `{problem_id, title}` |
| `trajectories` | SOLVER | CRITIC | `{trajectory_id, problem_id, run_tag}` |

Each queue has subdirectories: `pending/`, `processing/`, `done/`, `failed/`.
Messages are JSON files. State transitions use atomic `os.rename()` for POSIX safety.

### Agent Roles

| Agent | Type | Description |
|-------|------|-------------|
| CURATOR | One-shot (pure Python) | Loads KernelBench problems from HuggingFace, enqueues to `problems` |
| SOLVER | Queue loop | Dequeues problems, writes Triton kernels, verifies, enqueues trajectories |
| CRITIC | Queue loop | Dequeues trajectories, produces reflection cards to knowledge base |
| ORGANIZER | Scheduled (5 min) | Synthesizes knowledge cards from recent trajectories + reflections |
| INSIGHT_FINDER | Scheduled (10 min) | Detects cross-cutting meta-patterns across trajectories |

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
- **Shared data** (`problems/`, `cards/`, `lance/`, `queues/`) at env level, persists across runs
- **Per-run data** under `<run_tag>/<agent_name>/`, one JSON per entity
- **DuckDB** as query engine over JSON files (no persistent DB)
- **LanceDB** for vector embeddings (semantic search over cards)

## Design Rules

- **Tool-mediated verification**: The SOLVER must always use the `verifier` tool for correctness and performance checks. It must never attempt its own verification — no manual testing, no SSH to GPU hosts, no writing benchmark scripts, no running code locally. The verifier is the single source of truth.

## Key Patterns

- Custom tools: `@tool` decorator + `create_sdk_mcp_server` from `claude-agent-sdk`
- Tool references: `mcp__<server-key>__<tool-name>` in `allowed_tools`
- Data models: Pydantic v2 in `agenix/storage/models.py`
- Storage: JSON files (filesystem) + DuckDB (query) + LanceDB (vector)
- Knowledge: `tools/knowledge/baseline/` (KnowledgeStore, LanceDB index, embedder)
- Embeddings: `sentence-transformers` (local CPU) or remote Qwen3-Embedding-8B (GPU)
- Agents: Independent async processes communicating via filesystem queues + shared knowledge base
