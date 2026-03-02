# Reflection: System Design

## Async Queue-Based Architecture

Agents operate independently in separate processes, communicating via filesystem
queues and a shared knowledge base.

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

| Queue | Producer | Consumer | Payload |
|-------|----------|----------|---------|
| `problems` | CURATOR | SOLVER | `{problem_id, title}` |
| `experiences` | SOLVER | CRITIC | `{problem_id, experience_ids: [...]}` (batch) or `{experience_id, problem_id}` (single) |
| `reflections` | CRITIC | (future) | `{card_id}` |

### Parallel Solver

The solver supports `--parallel N` to run N instances per problem concurrently.
Each thread gets its own `ClaudeRunner` (with isolated `ToolRegistry` and MCP
servers) via a `runner_factory` callable. Knowledge retrieval is done once and
shared across all N runs.

```
CURATOR → [problems] → SOLVER (--parallel N)
                            │
                   ThreadPoolExecutor(N)
                   ╱        │        ╲
              solver#1  solver#2  solver#3
                 │          │         │
              exp_1      exp_2     exp_3
                   ╲        │        ╱
                    collect results
                            │
                            ▼
                    [experiences]
                    payload: {problem_id, experience_ids: [...]}
                            │
                            ▼
                         CRITIC
                   (batch variant for N>1,
                    comparative analysis)
```

### Agent Types

| Agent | Loop Type | Description |
|-------|-----------|-------------|
| CURATOR | One-shot | Pure Python KernelBench loader (no LLM) |
| SOLVER | QueueAgentLoop | Polls problems queue, writes Triton kernels |
| CRITIC | QueueAgentLoop | Polls experiences queue, produces reflection cards |
| ORGANIZER | ScheduledAgentLoop (5 min) | Synthesizes knowledge from recent data |
| INSIGHT_FINDER | ScheduledAgentLoop (10 min) | Cross-cutting meta-pattern detection |

### Problem Source: KernelBench

270 PyTorch GPU kernel problems from HuggingFace (`ScalingIntelligence/KernelBench`).
Each problem contains reference PyTorch code; solver writes Triton kernel replacements.

## Design Principles

### Tool-Mediated Verification

The SOLVER must **always** use the `verifier` tool for correctness and performance
checks. It must never attempt its own verification — no manual testing, no SSH to
GPU hosts, no writing benchmark scripts, no running code locally. The verifier is
the single source of truth for whether a solution compiles, is correct, and how it
performs relative to the reference.

**Rationale**: Self-verification is unreliable, wastes turns/cost exploring
infrastructure code, and produces results that aren't recorded in the system's
structured data. The verifier provides a standardized, reproducible evaluation
that feeds back into the learning loop.

## Data Layout

All data is stored as JSON files in a structured filesystem. DuckDB is used as
a query engine over these files (no persistent database).

```
~/.reflection/                             ← reflection_data_root
├── prod/                                  ← reflection_env
│   ├── problems/                          ← shared across runs
│   │   ├── <problem_id>.json
│   │   └── ...
│   ├── cards/                             ← shared across runs (all card types)
│   │   ├── <card_id>.json                 ← knowledge, insight, and reflection cards
│   │   └── ...
│   ├── lance/                             ← LanceDB vector index (shared)
│   │   └── cards.lance/
│   ├── queues/                            ← message queues (shared)
│   │   ├── problems/                      ← CURATOR → SOLVER
│   │   │   ├── pending/<message_id>.json
│   │   │   ├── processing/<message_id>.json
│   │   │   ├── done/<message_id>.json
│   │   │   └── failed/<message_id>.json
│   │   └── experiences/                   ← SOLVER → CRITIC
│   │       ├── pending/
│   │       ├── processing/
│   │       ├── done/
│   │       └── failed/
│   ├── experiences/                       ← shared across runs
│   │   ├── solver/
│   │   │   └── <experience_id>.jsonl      ← solver conversation logs
│   │   ├── critic/
│   │   │   └── ...
│   │   └── <agent_name>/
│   │       └── ...
│   ├── logs/                              ← agent log files
│   │   └── <agent_name>_<timestamp>.log
├── int/
│   └── ...
└── test_zhenchen/
    └── ...
```

### Storage Rules

- **Shared data** (`problems/`, `cards/`, `experiences/`, `lance/`) lives at the env level, persists across runs
- **Each JSON file** is a serialized Pydantic model (via `.model_dump(mode="json")`)
- **DuckDB queries** scan JSON files on demand: `read_json_auto('problems/*.json')`
- **LanceDB** stores vector embeddings for semantic search over cards
- **No persistent database** — the filesystem *is* the database

### Immutable Storage

The card storage system is **conceptually immutable** to guarantee full
traceability and lineage. Once a card is created, its content is never
modified and it is never deleted.

- **No modification**: Card content (`title`, `content`, `source_refs`) is
  fixed at creation. Any update creates a new card.
- **No deletion**: Cards are never removed from the filesystem. They are
  archived, which removes them from the vector index but preserves them
  on disk for lineage queries.
- **Superseded vs Archived**: Two distinct non-active statuses:
  - `SUPERSEDED`: Card was replaced by revision, merge, or split. The
    lineage event on the old card records which new card replaced it.
  - `ARCHIVED`: Card was manually retired. No successor exists. Used when
    a card is no longer relevant (outdated, deprecated).

## Knowledge Card Lifecycle

### Card Model

All card types (reflection, knowledge, insight) use a single unified `Card`
model. Type-specific behavior is driven by the `card_type` string field.

```
Card
├── card_id: str (ULID)
├── card_type: "reflection" | "knowledge" | "insight"
├── title, content, code_snippet
├── experience_ids: [str]              ← experiences that informed this card
├── tags: [str]                        ← keyword tags for search
├── applicability, limitations         ← when/how to apply, caveats
├── status: active | superseded | archived
├── source_refs: [{id, type}]         ← typed references (experience, card)
└── lineage: [LineageEvent]           ← append-only event log
      ├── operation: create | revise | merge | split | supersede | archive
      ├── timestamp, agent
      ├── description                 ← free text (e.g. "Merged from c1, c2")
      └── source_refs                 ← new sources added by this event
```

### Lifecycle Diagram

```
                    CREATE
                      │
                      ▼
                   ACTIVE ──────────────────────────┐
                   ╱  │  ╲                          │
              REVISE MERGE SPLIT               ARCHIVE
               ╱      │      ╲                      │
              ▼       ▼       ▼                     ▼
         SUPERSEDED  SUPERSEDED  SUPERSEDED     ARCHIVED
              │       │       │
              ▼       ▼       ▼
         new ACTIVE  new ACTIVE  new ACTIVE(s)
```

### Operations

All operations are in `agenix/storage/lineage.py`. Cards are never modified
in-place — revise/merge/split always produce NEW cards and supersede originals.

| Operation | Effect | Who |
|-----------|--------|-----|
| **CREATE** | New ACTIVE card with source_refs linking to experiences | CRITIC, ORGANIZER, INSIGHT_FINDER |
| **REVISE** | Old → SUPERSEDED. New card inherits source_refs + lineage | ORGANIZER |
| **MERGE** | N source cards → all SUPERSEDED. New card collects all source_refs | ORGANIZER |
| **SPLIT** | Original → SUPERSEDED. N new ACTIVE cards, each with subset | ORGANIZER |
| **ARCHIVE** | Card → ARCHIVED. Removed from LanceDB, kept on filesystem | Any agent |

### Card Producers

| Agent | Card Type | Trigger |
|-------|-----------|---------|
| CRITIC | reflection | Analyzes solver experiences (single or batch comparative) |
| ORGANIZER | knowledge | Periodic synthesis from recent reflections + experiences |
| INSIGHT_FINDER | insight | Periodic cross-cutting meta-pattern detection |

### Knowledge Tools (MCP)

8 individual MCP tools in `tools/knowledge/baseline/logic.py`:

| Tool | Description |
|------|-------------|
| `knowledge_search` | Semantic search over cards via LanceDB |
| `knowledge_list` | List cards by type, status, or tag |
| `knowledge_get` | Fetch a single card by ID |
| `knowledge_create` | Create a new card with lineage |
| `knowledge_revise` | Revise a card (old → superseded, new created) |
| `knowledge_merge` | Merge N cards into one (all → superseded) |
| `knowledge_split` | Split one card into N (original → superseded) |
| `knowledge_archive` | Archive a card (removed from vector index) |

### Reverse Lookups

- `find_cards_by_source(source_id)` → cards referencing that source
- `get_source_experiences(card)` → experience IDs from source_refs
- `get_source_reflections(card)` → reflection card IDs from source_refs

## Legend

```
AGENT           Agent (ALL UPPER CASE)
[DataName]      Data flowing between components
<ToolName>      Tool used by an agent
```
