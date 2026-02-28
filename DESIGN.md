# Reflection: System Design

## Autonomous Loop (Pipeline Flow)

```
                              CURATOR
                                 │
                                 ▼
                            [Problems]
                                 │
                                 ▼
   <Verifier> ◀───────────── SOLVER ◀──────────────── <Retriever>
        │                     ▲  │                          ▲
        │                     │  │                          │
        └─────────────────────┘  │                          │
               (loop)            ▼                          │
                            [Trajectory]                    │
                                 │                          │
                                 ▼                          │
                             REFLECTOR                      │
                                 │                          │
                                 ▼                          │
                          [Understandings]                  │
                                 │                          │
                    ┌────────────┴────────────┐             │
                    │                         │             │
                    ▼                         ▼             │
               ORGANIZER                INSIGHT_FINDER      │
                    │                         │             │
             [KnowledgeCards]           [InsightCards]      │
                    │                         │             │
                    │                         │             │
                    └──────────┬──────────────┘             │
                               ▼                            │
                       [Knowledge Base] ────────────────────┘
```

## Data Layout

```
~/.reflection/                             ← reflection_data_root
├── prod/                                  ← reflection_env
│   ├── reflection.db                      ← shared across runs
│   ├── chroma/                            ← shared across runs
│   ├── run_20260228_143000/               ← run_tag
│   │   ├── curator/                       ← per-agent outputs
│   │   ├── solver/
│   │   └── ...
│   └── run_20260228_150000/
│       └── ...
├── int/
│   └── ...
└── test_zhenchen/
    └── ...
```

## Legend

```
AGENT           Agent (ALL UPPER CASE)
[DataName]      Data flowing between components
<ToolName>      Tool used by an agent
```
