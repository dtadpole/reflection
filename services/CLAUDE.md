# Services

Remote service deployment and management for GPU workloads.

## Service Structure

Services follow a **variant pattern**: `services/<service_name>/<variant>/`. This mirrors the agent and tool folder patterns.

```
services/
├── __init__.py
├── models.py                      # Shared Pydantic models
├── deploy.py                      # SSH deployer
├── health.py                      # Health checker
├── kb_eval/                       # kbEval service
│   ├── __init__.py
│   └── baseline/                  # baseline variant
│       ├── __init__.py
│       ├── __main__.py            # python -m services.kb_eval.baseline
│       ├── client.py              # HTTP client
│       ├── server.py              # FastAPI server
│       ├── worker.py              # Subprocess worker
│       ├── util.py                # Eval utilities
│       └── deploy/                # Deployment artifacts
│           ├── requirements.txt
│           └── kb-eval.service
├── text_embedding/                # Text embedding service
│   ├── __init__.py
│   └── baseline/                  # baseline variant (Qwen3-Embedding-8B)
│       ├── __init__.py
│       ├── __main__.py            # python -m services.text_embedding.baseline
│       ├── client.py              # HTTP client
│       ├── server.py              # FastAPI server
│       └── deploy/
│           └── requirements.txt
├── reranker/                      # Cross-encoder reranker service
│   ├── __init__.py
│   ├── CLAUDE.md                  # Service contract
│   └── baseline/                  # baseline variant (Qwen3.5-27B via vLLM)
│       ├── __init__.py
│       ├── __main__.py            # python -m services.reranker.baseline
│       ├── client.py              # HTTP client
│       ├── server.py              # FastAPI server (vLLM wrapper)
│       └── deploy/
│           ├── requirements.txt
│           ├── reranker.service   # systemd unit for FastAPI
│           └── reranker-vllm.service  # systemd unit for vLLM
└── ssh_tunnel/                    # SSH tunnel service (infrastructure)
    ├── __init__.py
    ├── tunnel.py                  # TunnelStatus, check_port, get_manager
    ├── mac/manager.py             # LaunchdTunnelManager
    └── linux/manager.py           # SystemdTunnelManager (stub)
```

When adding a new service variant:
1. Create `services/<service_name>/<variant>/` with `__init__.py`
2. Implementation files live inside the variant directory
3. Imports use the full path: `from services.<service_name>.<variant>.module import X`
4. Shared models stay in `services/models.py` (not inside any variant)

## Deployment Patterns

- **Always use `uv`** to manage Python environments on remote servers (not `python3 -m venv` + `pip`)
  - `uv` is available at `~/.local/bin/uv` on remotes
  - `python3-venv` package is not installed on remote hosts
- **Secret keys** stored under `~/.keys/` (both local and remote)
- **Service base directory**: `~/.reflection/services/<service-name>/` on remote hosts
- **Venv location**: `~/.reflection/services/<service-name>/.venv/`
- **systemd user units** for service lifecycle (`systemctl --user`, `loginctl enable-linger`)
- **ExecStart must be a single line** — systemd does not support backslash continuations
- **asyncssh SFTP cannot expand `~`** — always resolve `$HOME` first for remote paths

## Remote Hosts

Configured in `config/hosts.yaml`. SSH aliases (`_one`, `_two`) defined in `~/.ssh/config`.

## CLI Commands

```bash
reflection services deploy <name>           # Deploy kbEval via systemd
reflection services stop <name>             # Stop kbEval
reflection services deploy-embedding <name> # Deploy text-embedding via systemd
reflection services stop-embedding <name>   # Stop text-embedding
reflection services deploy-reranker <name>  # Deploy reranker (vLLM + FastAPI) via systemd
reflection services stop-reranker <name>    # Stop reranker
reflection services status                  # Health-check all endpoints
reflection services health <name>           # Detailed health (all services) + systemd status
reflection services logs <name>             # View kbEval journal logs
reflection services logs-embedding <name>   # View text-embedding journal logs
reflection services logs-reranker <name>    # View reranker journal logs (both units)
```

## SSH Tunnels

Persistent SSH tunnels for port forwarding to remote services. Managed via launchd (Mac) or systemd (Linux). Config lives in `config/tunnels.yaml`.

```bash
reflection services tunnel start             # Start all tunnels
reflection services tunnel start _one        # Start single tunnel
reflection services tunnel stop              # Stop all tunnels
reflection services tunnel stop _one         # Stop single tunnel
reflection services tunnel restart           # Restart all (after config change)
reflection services tunnel restart _one      # Restart single tunnel
reflection services tunnel status            # Show tunnel status
```

Once tunnels are running, all HTTP access (health checks, tests, eval) goes through `localhost:<port>` — no manual SSH tunnels needed.

## Testing

Integration tests live in `tests/integration/test_services.py`. Run with `make test-service`.

Tests are organized in two layers:

1. **Service infrastructure** — SSH connectivity, systemd unit management, journal log access.
   Tests the deployment machinery independent of any specific service.

2. **Per-service tests** — each service type gets its own test class:
   - `TestKbEvalHealth` — health endpoint, GPU device detection, pending request count
   - `TestKbEvalPyTorch` — eval round-trips with correct/wrong PyTorch code, eval_ref benchmark

### Prerequisites

- SSH access to remote hosts (configured in `config/hosts.yaml`)
- SSH tunnels running: `reflection services tunnel start`
- kbEval deployed and running on the target host

### Test ordering

`make test-unit` → `make test-service` → `make test-integration`

`test-service` depends on `test-unit`; `test-integration` depends on `test-service`.
