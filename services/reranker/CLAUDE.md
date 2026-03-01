# Reranker Service

Zero-shot cross-encoder reranker using Qwen3-32B via SGLang.

## Architecture

Two-process design on `_two`:
- **reranker-baseline** (port 42984): SGLang serving Qwen3-32B. Internal only, not tunneled.
- **reranker** (port 42983): FastAPI wrapper that constructs prompts, calls SGLang, extracts scores. Exposed via SSH tunnel.

## How Scoring Works

Uses the yes/no logit trick:
1. For each (query, document) pair, construct a ChatML prompt asking "is this relevant? yes/no"
2. Call SGLang with `max_tokens=1, logprobs=5` (OpenAI-compatible API)
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
  "model": "Qwen/Qwen3-32B"
}
```

### GET /health

Returns `ServiceHealth` with `devices` indicating SGLang backend status (`sglang:ok`, `sglang:error`, `sglang:unreachable`).

## CLI Commands

```bash
reflection services deploy-reranker _two    # Deploy both SGLang + FastAPI
reflection services stop-reranker _two      # Stop both services
reflection services logs-reranker _two      # View logs (both units)
reflection services health _two             # Includes reranker status
```

## Configuration

Server config in `agenix/config.py`: `RerankerServerConfig` (port, vllm_port, model_name, device).
Client config: `RerankerClientConfig` (base_url, timeout, retry_count, retry_interval).
Host config in `config/hosts.yaml`, tunnel in `config/tunnels.yaml`.

## Deployment Notes

- **Engine**: SGLang (not vLLM) — vLLM pre-built wheels don't support Blackwell (sm_120) GPUs
- **SGLang flags**: `--disable-cuda-graph --mem-fraction-static 0.80 --context-length 24576`
- **flashinfer**: SGLang depends on pre-release flashinfer; use `--prerelease=allow` with uv
- **PATH in systemd**: The unit sets `Environment=PATH=...` so ninja (JIT compilation) is found
