# Knowledge Tool

The knowledge tool manages the knowledge base — the persistent store of cards (knowledge, insight, reflection) that accumulates across pipeline iterations. It is the central data layer connecting agents to stored knowledge.

## Design Objectives

1. **Dual-store consistency**: Every card operation must keep the filesystem (JSON + DuckDB) and vector database (LanceDB) in sync. Active cards exist in both stores; superseded/archived cards exist only on the filesystem.

2. **Immutable card history**: Cards are never modified after creation. Revise, merge, and split all produce new cards and supersede the originals. The full lineage chain is preserved for traceability.

3. **Source lineage preservation**: Every card tracks which trajectories (and other entities) contributed to its creation via `source_refs`. Lineage operations (revise, merge, split) must propagate source refs from predecessors to successors.

4. **Predecessor lineage**: Every card tracks its direct ancestors via `predecessor_ids`. This forms a DAG that can be traversed to reconstruct the full evolution history of any piece of knowledge.

## Card Lifecycle Operations

### Create
- Card is saved to filesystem (status=ACTIVE) and embedded in LanceDB.
- `record_creation()` sets initial `source_refs` and appends a CREATE lineage event.
- DuckDB sees the card with status "active".

### Revise
- Old card: status=SUPERSEDED, `superseded_by` points to new card, removed from LanceDB.
- New card: status=ACTIVE, `predecessor_ids` includes old card, added to LanceDB.
- Source refs are inherited from old card + any new refs added.
- REVISE event on new card, SUPERSEDE event on old card.

### Merge
- All source cards: status=SUPERSEDED, `superseded_by` points to merged card, removed from LanceDB.
- Merged card: status=ACTIVE, `predecessor_ids` includes all source card IDs, added to LanceDB.
- Source refs are collected from all source cards (deduplicated).
- MERGE event on merged card (with `merged_card_ids`), SUPERSEDE events on each source.

### Split
- Original card: status=SUPERSEDED, `superseded_by` points to first child, removed from LanceDB.
- Child cards: status=ACTIVE, `predecessor_ids` includes original, added to LanceDB.
- Source refs are partitioned per child (caller decides which refs go where).
- SPLIT event on each child (with `split_from_card_id`), SUPERSEDE event on original.

### Archive
- Card: status=ARCHIVED, removed from LanceDB, kept on filesystem.
- ARCHIVE lineage event recorded.
- No successor card is created.

## Consistency Invariants

These must hold after every operation:

1. **LanceDB row count == FS active card count** (for any domain/type filter)
2. **DuckDB sees all statuses** — active, superseded, and archived cards are all queryable via `fs.query_cards()`
3. **Semantic search only returns active cards** — superseded and archived cards must never appear in `store.search()` results
4. **All cards persist on filesystem** — nothing is ever deleted, only status changes
5. **`superseded_by` is always set** on SUPERSEDED cards (for revise, merge, and split)

## Embedder Variants

The knowledge store accepts any embedder implementing the `embed(texts) -> ndarray` and `embed_one(text) -> ndarray` interface:

- **`Embedder`** (local): `sentence-transformers` model on CPU. Default: `all-MiniLM-L6-v2` (384-dim). Used for dev/test.
- **`RemoteEmbedder`** (remote GPU): Synchronous HTTP adapter calling the `text_embedding` service. Default: `Qwen3-Embedding-8B` (4096-dim) on `_two`. Used for production.

## Module Structure

```
tools/knowledge/baseline/
├── store.py      # KnowledgeStore — composite FS + LanceDB store
├── embedder.py   # Embedder (local) + RemoteEmbedder (remote GPU)
├── index.py      # LanceIndex — LanceDB vector index wrapper
├── logic.py      # create_tool() factory — MCP tool interface
├── tool.md       # Tool description and schema
├── config.yaml   # name="knowledge", variant="baseline"
└── __init__.py
```

## Dependencies

- `store.py` imports from `agenix.config`, `agenix.storage.fs_backend`, `agenix.storage.models`
- `index.py` imports from `agenix.config` (StorageConfig for default lance_path)
- `embedder.py` imports from `agenix.config` (EmbedderConfig, TextEmbeddingClientConfig)
- Lineage operations live in `agenix/storage/lineage.py` (not in this tool)

## Testing

```bash
# Integration tests (requires text-embedding service on _two + SSH tunnels)
uv run pytest tests/integration/test_knowledge_base.py -v -s

# Tests verify all 5 operations with:
# - FS status checks (get_card + status assertion)
# - LanceDB presence/absence checks (direct table query)
# - DuckDB query checks (query_cards with status assertion)
# - Source lineage preservation (source_refs round-trip)
# - Predecessor lineage (predecessor_ids, superseded_by)
# - Semantic search correctness (active cards found, superseded/archived excluded)
```
