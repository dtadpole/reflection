# Solver Tools

## knowledge_retriever
Query the knowledge base for relevant cards.
- Input: `{"query": "search query", "top_k": 5}`
- Output: `{"cards": [{"title": "...", "content": "...", "score": 0.85}]}`

## verifier
Evaluate a generated GPU kernel against a reference implementation.
- Input: `{"reference_code": "...", "generated_code": "...", "code_type": "triton"}`
- Output: `{"compiled": true, "correctness": true, "runtime": 0.42, "metadata": {...}, "runtime_stats": {...}}`
- code_type: "triton" | "cuda" | "pytorch" (default: "triton")
