# Retriever (base)

## Description
Query the knowledge base for relevant cards by semantic similarity using
LanceDB dense vector retrieval with sentence-transformers embeddings.

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
  - score (float): Similarity score (0-1, higher is better)

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
      "score": 0.8523
    }
  ]
}
```
