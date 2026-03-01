# Services

Remote service deployment and management for GPU workloads.

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
reflection services deploy <name>    # Deploy & start via systemd
reflection services stop <name>      # Stop service
reflection services status           # Health-check all endpoints
reflection services health <name>    # Detailed health + systemd status
reflection services logs <name>      # View journal logs
```
