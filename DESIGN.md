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
| `experiences` | SOLVER | CRITIC | `{experience_id, problem_id}` |

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
│   │   │   └── <experience_id>.jsonl      ← solver experiences
│   │   ├── critic/
│   │   │   └── ...
│   │   └── <agent_name>/
│   │       └── ...
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
- **Predecessor chain**: Every new card produced by revision, merge, or split
  records `predecessor_ids` pointing to the cards it was derived from. This
  enables walking the full ancestry chain back to the original sources.
- **Superseded vs Archived**: Two distinct non-active statuses:
  - `SUPERSEDED`: Card was replaced by revision, merge, or split. Always has
    `superseded_by` pointing to the replacement card. Enables walking the
    chain forward to find the current version.
  - `ARCHIVED`: Card was manually retired. No successor exists. Used when
    a card is no longer relevant (outdated, deprecated).

## Data Lineage

Every knowledge card carries an append-only lineage log that records how it was
created and evolved. This enables tracing any card back to its source
experiences, reflection cards, and predecessor cards.

### Lineage Model

```
Card
├── status: active | superseded | archived
├── source_refs: [{id, type}]          ← typed references (experience, reflection, card)
├── source_ids: [str]                  ← kept in sync with source_refs for compat
├── predecessor_ids: [str]             ← cards that were inputs (merge/split/revision)
├── superseded_by: str | null          ← quick lookup for replacement card
└── lineage: [LineageEvent]            ← append-only event log
      ├── operation: create | revise | merge | split | supersede | archive
      ├── timestamp, agent, run_tag
      ├── source_refs                  ← new sources added by this event
      ├── from_version                 ← for revise/merge: previous version
      ├── merged_card_ids              ← for merge: cards absorbed
      ├── split_from_card_id           ← for split: parent card
      └── superseded_by                ← for supersede: replacement card
```

### Operations

All operations follow the immutable storage principle: existing cards are
never modified (only their status/superseded_by fields are updated for
archiving). New content always goes into a new card.

- **Create**: Card created from experience + reflection cards → `source_refs` populated
- **Revise**: NEW card created with updated content. Old card superseded (`superseded_by`
  → new card, new card `predecessor_ids` → old card). New card inherits `source_refs`
  from old + any new sources.
- **Merge**: NEW card created from multiple source cards. All sources superseded
  (`superseded_by` → new card, new card `predecessor_ids` → source cards). New card gets
  union of `source_refs`.
- **Split**: NEW cards created from original. Original superseded. Each child gets
  caller-specified `source_refs`. `predecessor_ids = [original_card_id]`
- **Archive**: Card manually retired → status set to `archived`

### Reverse Lookups

- `find_cards_by_source(source_id)` → cards referencing that source
- `get_source_experiences(card)` → experience IDs from source_refs
- `get_card_ancestry(card, all_cards)` → recursive predecessor chain

## Legend

```
AGENT           Agent (ALL UPPER CASE)
[DataName]      Data flowing between components
<ToolName>      Tool used by an agent
```
