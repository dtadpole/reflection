# Knowledge

## Description
Manage the knowledge base: add, search, list, and retrieve knowledge cards.
Combines filesystem storage (JSON + DuckDB) with LanceDB vector index for
semantic search over knowledge, insight, and reflection cards.

## Input Schema
- action (str, required): Operation to perform (search/add/list/get)
- query (str, optional): Search query text (required for search)
- top_k (int, optional): Number of search results (default: 5)
- card_type (str, optional): Filter by type (knowledge/reflection/insight)
- domain (str, optional): Filter by domain
- card_id (str, optional): Card ID (required for get)
- card (object, optional): Full card object (required for add)

## Output Schema
- cards (list): Matching cards with fields:
  - card_id (str): Unique card identifier
  - title (str): Card title
  - content (str): Card content
  - card_type (str): Type of card
  - score (float): Similarity score (0-1, for search results)

## Examples
Search:
```json
{
  "action": "search",
  "query": "matrix multiplication optimization",
  "top_k": 3,
  "card_type": "knowledge"
}
```

Get:
```json
{
  "action": "get",
  "card_id": "card_01HXYZ..."
}
```
