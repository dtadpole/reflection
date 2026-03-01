# Retriever Tool

The retriever tool provides semantic search over the knowledge base, enabling agents to find relevant cards by natural language query.

## Design Pattern

The retriever is a thin MCP wrapper around `KnowledgeStore.search()`. It converts raw distance scores to similarity scores (1.0 - distance) and formats results for agent consumption.

## Contract

### Input
- `query` (str, required): Natural language search query
- `top_k` (int, optional): Max results to return (default: 5)
- `card_type` (str, optional): Filter by card type (knowledge/reflection/insight)

### Output
- `cards` (list): Ranked results, most relevant first, each with:
  - `card_id`, `title`, `content`, `card_type`, `score` (0-1, higher = better)

### Guarantees
- Results are ordered by descending similarity score
- Only ACTIVE cards appear in results (superseded/archived cards are excluded)
- Invalid `card_type` values return an error (not silently ignored)
- Empty query returns an error

## Variant: `baseline`

- **Backend**: `KnowledgeStore` with LanceDB vector search
- **Embedder**: Whatever embedder the store was initialized with (local or remote)
- **MCP tool name**: `knowledge_retriever`
- **Dependencies**: `knowledge_store: KnowledgeStore` injected via `create_tool()`

## Variant: `rerank`

- **Backend**: Two-stage pipeline — `KnowledgeStore` (dense retrieval) + `RerankerClient` (cross-encoder)
- **Stage 1**: Retrieve 5*K candidates via embedding search
- **Stage 2**: Rerank candidates via cross-encoder (Qwen3-32B), return top K
- **MCP tool name**: `knowledge_retriever` (same as baseline — only one variant is registered)
- **Dependencies**: `knowledge_store: KnowledgeStore` + `reranker_client: RerankerClient` injected via `create_tool()`
- **Auto-selected**: When `config.services.endpoints` is configured (reranker endpoint available)
- **Scores**: Reranker relevance scores (not embedding distance), 0-1

## Testing

```bash
# Unit tests for rerank variant (mocked store + reranker)
uv run pytest tests/unit/test_retriever_rerank.py -v

# Unit-style integration tests for baseline (local embedder, no remote service needed)
uv run pytest tests/integration/test_retriever_tool.py -v -s

# Tests cover: query, empty query, top_k, type filter, invalid type,
# semantic quality (correct ranking), lineage-aware exclusion
```
