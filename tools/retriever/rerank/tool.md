# Retriever (rerank)

## Description
Query the knowledge base for relevant cards using a two-stage pipeline:
dense vector retrieval to fetch candidates, then cross-encoder reranking
(Qwen3-32B) for precise relevance scoring.

## Input Schema
- query (str, required): Search query text
- top_k (int, optional): Number of results (default: 5)
- card_type (str, optional): Filter by type (knowledge/reflection/insight)

## Output Schema
- cards (list): Matching cards with fields:
  - card_id (str): Unique card identifier
  - title (str): Card title
  - content (str): Card content
  - card_type (str): Type of card
  - score (float): Reranker relevance score (0-1, higher is better)

## Examples
Input:
```json
{
  "query": "matrix multiplication optimization",
  "top_k": 3,
  "card_type": "knowledge"
}
```

Output:
```json
{
  "cards": [
    {
      "card_id": "card_01HXYZ...",
      "title": "Tiled Matrix Multiplication in Triton",
      "content": "Use shared memory tiles...",
      "card_type": "knowledge",
      "score": 0.9512
    }
  ]
}
```
