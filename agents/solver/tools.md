# Solver Tools

## code_executor
Execute Python code in a sandboxed subprocess.
- Input: `{"code": "python code string"}`
- Output: `{"stdout": "...", "stderr": "...", "exit_code": 0}`
- Timeout: 30 seconds

## knowledge_retriever
Query the knowledge base for relevant cards.
- Input: `{"query": "search query", "top_k": 5}`
- Output: `{"cards": [{"title": "...", "content": "...", "score": 0.85}]}`
