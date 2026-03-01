# Reranker Service

Zero-shot cross-encoder reranker using Qwen3.5-27B via vLLM.

## Architecture

Two-process design on `_two`:
- **reranker-vllm** (port 42984): vLLM serving Qwen3.5-27B. Internal only, not tunneled.
- **reranker** (port 42983): FastAPI wrapper that constructs prompts, calls vLLM, extracts scores. Exposed via SSH tunnel.

## How Scoring Works

Uses the yes/no logit trick:
1. For each (query, document) pair, construct a ChatML prompt asking "is this relevant? yes/no"
2. Call vLLM with `max_tokens=1, logprobs=5`
3. Extract logprobs for "yes" and "no" tokens
4. Score = `exp(logprob_yes) / (exp(logprob_yes) + exp(logprob_no))`

## API

### POST /rank

```json
{
  "query": "optimize matrix multiply",
  "documents": ["Use tiling for cache locality", "Hello world program"],
  "instruction": "Given the query, determine if the document is relevant."
}
```

Response:
```json
{
  "scores": [0.95, 0.02],
  "model": "Qwen/Qwen3.5-27B"
}
```

### GET /health

Returns `ServiceHealth` with `devices` indicating vLLM backend status (`vllm:ok`, `vllm:error`, `vllm:unreachable`).

## CLI Commands

```bash
reflection services deploy-reranker _two    # Deploy both vLLM + FastAPI
reflection services stop-reranker _two      # Stop both services
reflection services logs-reranker _two      # View logs (both units)
reflection services health _two             # Includes reranker status
```

## Configuration

Server config in `agenix/config.py`: `RerankerServerConfig` (port, vllm_port, model_name, device).
Client config: `RerankerClientConfig` (base_url, timeout, retry_count, retry_interval).
Host config in `config/hosts.yaml`, tunnel in `config/tunnels.yaml`.
