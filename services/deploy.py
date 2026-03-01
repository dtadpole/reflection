"""SSH-based deployer for remote services using systemd user units."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from agenix.config import ServiceEndpoint, ServicesConfig
from services.models import ServiceHealth, ServiceStatus

logger = logging.getLogger(__name__)

# Service code files to upload
_KB_EVAL_FILES = [
    "services/__init__.py",
    "services/models.py",
    "services/kb_eval/__init__.py",
    "services/kb_eval/baseline/__init__.py",
    "services/kb_eval/baseline/__main__.py",
    "services/kb_eval/baseline/server.py",
    "services/kb_eval/baseline/worker.py",
    "services/kb_eval/baseline/util.py",
]

_REQUIREMENTS_FILE = "services/kb_eval/baseline/deploy/requirements.txt"

_UNIT_NAME = "kb-eval"
_REMOTE_BASE = "~/.reflection/services/kb_eval"

# Text embedding service
_TEXT_EMBEDDING_FILES = [
    "services/__init__.py",
    "services/models.py",
    "services/text_embedding/__init__.py",
    "services/text_embedding/baseline/__init__.py",
    "services/text_embedding/baseline/__main__.py",
    "services/text_embedding/baseline/server.py",
    "services/text_embedding/baseline/client.py",
]
_TEXT_EMBEDDING_REQUIREMENTS = "services/text_embedding/baseline/deploy/requirements.txt"
_TEXT_EMBEDDING_UNIT_NAME = "text-embedding"

# Reranker service (FastAPI + vLLM backend)
_RERANKER_FILES = [
    "services/__init__.py",
    "services/models.py",
    "services/reranker/__init__.py",
    "services/reranker/baseline/__init__.py",
    "services/reranker/baseline/__main__.py",
    "services/reranker/baseline/server.py",
    "services/reranker/baseline/client.py",
]
_RERANKER_REQUIREMENTS = "services/reranker/baseline/deploy/requirements.txt"
_RERANKER_UNIT_NAME = "reranker"
_RERANKER_VLLM_UNIT_NAME = "reranker-baseline"


def _render_text_embedding_unit(port: int, model: str, dimension: int, device: str) -> str:
    """Render the systemd unit file for text embedding service."""
    exec_start = (
        "%h/.reflection/services/text_embedding/.venv/bin/python"
        " -m services.text_embedding.baseline"
        f" --host 0.0.0.0 --port {port}"
        f" --model {model}"
        f" --dimension {dimension}"
        f" --device {device}"
    )
    return textwrap.dedent(f"""\
        [Unit]
        Description=Text embedding server (Qwen3-Embedding-8B)
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory=%h/.reflection/services/text_embedding
        ExecStart={exec_start}
        Restart=always
        RestartSec=5

        [Install]
        WantedBy=default.target
    """)


def _render_unit(port: int, devices: str) -> str:
    """Render the systemd unit file with endpoint-specific values."""
    # ExecStart must be a single line — systemd doesn't support shell continuations
    exec_start = (
        "%h/.reflection/services/kb_eval/.venv/bin/python"
        " -m services.kb_eval.baseline"
        f" --host 0.0.0.0 --port {port}"
        f" --devices {devices}"
        " --data-root %h/.reflection"
    )
    return textwrap.dedent(f"""\
        [Unit]
        Description=kbEval GPU kernel evaluation server
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory=%h/.reflection/services/kb_eval
        ExecStart={exec_start}
        Restart=always
        RestartSec=5

        [Install]
        WantedBy=default.target
    """)


def _render_reranker_backend_unit(
    backend_port: int, model: str, device: str
) -> str:
    """Render the systemd unit file for the reranker SGLang backend."""
    exec_start = (
        "%h/.reflection/services/reranker/.venv/bin/python"
        f" -m sglang.launch_server"
        f" --model-path {model}"
        f" --port {backend_port}"
        " --dtype float16"
        " --context-length 24576"
        " --mem-fraction-static 0.80"
        " --disable-cuda-graph"
    )
    env_path = (
        "PATH=%h/.reflection/services/reranker/.venv/bin"
        ":/usr/local/bin:/usr/bin:/bin"
    )
    return textwrap.dedent(f"""\
        [Unit]
        Description=SGLang server for reranker baseline ({model})
        After=network.target

        [Service]
        Type=simple
        WorkingDirectory=%h/.reflection/services/reranker
        Environment={env_path}
        ExecStart={exec_start}
        Restart=always
        RestartSec=10

        [Install]
        WantedBy=default.target
    """)


def _render_reranker_unit(port: int, vllm_port: int, model: str) -> str:
    """Render the systemd unit file for the reranker FastAPI wrapper."""
    exec_start = (
        "%h/.reflection/services/reranker/.venv/bin/python"
        " -m services.reranker.baseline"
        f" --host 0.0.0.0 --port {port}"
        f" --vllm-url http://localhost:{vllm_port}"
        f" --model {model}"
    )
    return textwrap.dedent(f"""\
        [Unit]
        Description=Reranker FastAPI server (cross-encoder via SGLang)
        After=network.target {_RERANKER_VLLM_UNIT_NAME}.service

        [Service]
        Type=simple
        WorkingDirectory=%h/.reflection/services/reranker
        ExecStart={exec_start}
        Restart=always
        RestartSec=5

        [Install]
        WantedBy=default.target
    """)


class ServiceDeployer:
    """Deploys and manages services on remote machines via SSH + systemd."""

    def __init__(self, config: ServicesConfig) -> None:
        self._config = config

    async def deploy_kb_eval(self, endpoint: ServiceEndpoint) -> bool:
        """Deploy kbEval to a remote endpoint.

        Steps:
        1. Upload service code to ~/.reflection/services/kb_eval/
        2. Install systemd user unit
        3. Enable lingering (so service survives logout)
        4. Start the service

        Args:
            endpoint: Target endpoint with SSH credentials.

        Returns:
            True if deployment succeeded.
        """
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            # 1. Resolve remote home and create directory structure
            home_result = await conn.run("echo $HOME", check=True)
            remote_home = home_result.stdout.strip()
            remote_base = f"{remote_home}/.reflection/services/kb_eval"

            await conn.run(
                f"mkdir -p {remote_base}/services/kb_eval/baseline "
                f"{remote_home}/.config/systemd/user",
                check=True,
            )

            # 2. Upload service code + requirements
            project_root = Path(__file__).parent.parent
            async with conn.start_sftp_client() as sftp:
                for rel_path in _KB_EVAL_FILES:
                    local = project_root / rel_path
                    if not local.exists():
                        logger.warning("File not found: %s", local)
                        continue
                    remote = f"{remote_base}/{rel_path}"
                    await sftp.put(str(local), remote)

                # Upload requirements.txt
                req_local = project_root / _REQUIREMENTS_FILE
                if req_local.exists():
                    await sftp.put(str(req_local), f"{remote_base}/requirements.txt")

            # 3. Create venv and install dependencies (using uv)
            venv_path = f"{remote_base}/.venv"
            uv_bin = f"{remote_home}/.local/bin/uv"
            venv_check = await conn.run(f"test -f {venv_path}/bin/python")
            if venv_check.exit_status != 0:
                logger.info("Creating venv on %s ...", endpoint.name)
                await conn.run(f"{uv_bin} venv {venv_path}", check=True)
                logger.info("Installing dependencies on %s ...", endpoint.name)
                await conn.run(
                    f"{uv_bin} pip install -p {venv_path}/bin/python "
                    f"-r {remote_base}/requirements.txt",
                    check=True,
                )
                logger.info("Dependencies installed on %s", endpoint.name)
            else:
                # Venv exists — just update deps in case requirements changed
                logger.info("Updating dependencies on %s ...", endpoint.name)
                await conn.run(
                    f"{uv_bin} pip install -q -p {venv_path}/bin/python "
                    f"-r {remote_base}/requirements.txt",
                )

            # 5. Write systemd unit file
            server_cfg = self._config.kb_eval_server
            devices = ",".join(server_cfg.devices)
            unit_content = _render_unit(endpoint.kb_eval_port, devices)
            unit_path = f"{remote_home}/.config/systemd/user/{_UNIT_NAME}.service"
            r = await conn.run(f"cat > {unit_path} << 'UNIT_EOF'\n{unit_content}UNIT_EOF")
            if r.exit_status != 0:
                logger.error("Failed to write unit file: %s", r.stderr)
                conn.close()
                return False

            # 6. Enable lingering (service runs without active session)
            await conn.run("loginctl enable-linger $(whoami) 2>/dev/null || true")

            # 7. Reload systemd and start
            await conn.run("systemctl --user daemon-reload", check=True)
            await conn.run(f"systemctl --user enable {_UNIT_NAME}", check=True)
            await conn.run(f"systemctl --user restart {_UNIT_NAME}", check=True)

            # 8. Wait for service to start (torch import can take several seconds)
            import asyncio
            await asyncio.sleep(10)
            status_result = await conn.run(
                f"systemctl --user is-active {_UNIT_NAME}"
            )
            is_active = status_result.stdout.strip() == "active"

            conn.close()

            if is_active:
                logger.info(
                    "Deployed kbEval to %s (port %d, devices: %s)",
                    endpoint.name, endpoint.kb_eval_port, devices,
                )
            else:
                logger.error(
                    "kbEval deployed but not active on %s", endpoint.name
                )

            return is_active

        except Exception as e:
            logger.error("Deploy failed for %s: %s", endpoint.name, e)
            return False

    async def stop_kb_eval(self, endpoint: ServiceEndpoint) -> bool:
        """Stop kbEval on a remote endpoint.

        Args:
            endpoint: Target endpoint.

        Returns:
            True if stop command succeeded.
        """
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            await conn.run(f"systemctl --user stop {_UNIT_NAME}")
            conn.close()

            logger.info("Stopped kbEval on %s", endpoint.name)
            return True

        except Exception as e:
            logger.error("Stop failed for %s: %s", endpoint.name, e)
            return False

    async def status_kb_eval(self, endpoint: ServiceEndpoint) -> ServiceHealth:
        """Check if kbEval is running on a remote endpoint.

        Queries the HTTP health endpoint first, falls back to systemd status.

        Args:
            endpoint: Target endpoint.

        Returns:
            ServiceHealth with current status.
        """
        from services.kb_eval.baseline.client import KbEvalClient

        client = KbEvalClient(endpoint.kb_eval)
        try:
            return await client.health()
        except Exception:
            return ServiceHealth(
                name=endpoint.name,
                status=ServiceStatus.STOPPED,
                endpoint=endpoint.kb_eval.base_url,
            )

    async def logs_kb_eval(
        self, endpoint: ServiceEndpoint, lines: int = 50
    ) -> str:
        """Fetch recent journal logs for kbEval on a remote endpoint.

        Args:
            endpoint: Target endpoint.
            lines: Number of log lines to fetch.

        Returns:
            Log output string.
        """
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            result = await conn.run(
                f"journalctl --user -u {_UNIT_NAME} -n {lines} --no-pager"
            )
            conn.close()
            return result.stdout

        except Exception as e:
            return f"Failed to fetch logs for {endpoint.name}: {e}"

    async def systemd_status_kb_eval(self, endpoint: ServiceEndpoint) -> str:
        """Fetch systemctl status output for kbEval on a remote endpoint.

        Args:
            endpoint: Target endpoint.

        Returns:
            Status output string.
        """
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            result = await conn.run(
                f"systemctl --user status {_UNIT_NAME} --no-pager"
            )
            conn.close()
            return result.stdout

        except Exception as e:
            return f"Failed to get status for {endpoint.name}: {e}"

    # --- Text embedding methods ---

    async def deploy_text_embedding(self, endpoint: ServiceEndpoint) -> bool:
        """Deploy text embedding service to a remote endpoint."""
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            home_result = await conn.run("echo $HOME", check=True)
            remote_home = home_result.stdout.strip()
            remote_base = f"{remote_home}/.reflection/services/text_embedding"

            await conn.run(
                f"mkdir -p {remote_base}/services/text_embedding/baseline "
                f"{remote_home}/.config/systemd/user",
                check=True,
            )

            project_root = Path(__file__).parent.parent
            async with conn.start_sftp_client() as sftp:
                for rel_path in _TEXT_EMBEDDING_FILES:
                    local = project_root / rel_path
                    if not local.exists():
                        logger.warning("File not found: %s", local)
                        continue
                    remote = f"{remote_base}/{rel_path}"
                    await sftp.put(str(local), remote)

                req_local = project_root / _TEXT_EMBEDDING_REQUIREMENTS
                if req_local.exists():
                    await sftp.put(str(req_local), f"{remote_base}/requirements.txt")

            venv_path = f"{remote_base}/.venv"
            uv_bin = f"{remote_home}/.local/bin/uv"
            venv_check = await conn.run(f"test -f {venv_path}/bin/python")
            if venv_check.exit_status != 0:
                logger.info("Creating venv on %s ...", endpoint.name)
                await conn.run(f"{uv_bin} venv {venv_path}", check=True)
                logger.info("Installing dependencies on %s ...", endpoint.name)
                await conn.run(
                    f"{uv_bin} pip install -p {venv_path}/bin/python "
                    f"-r {remote_base}/requirements.txt",
                    check=True,
                )
                logger.info("Dependencies installed on %s", endpoint.name)
            else:
                logger.info("Updating dependencies on %s ...", endpoint.name)
                await conn.run(
                    f"{uv_bin} pip install -q -p {venv_path}/bin/python "
                    f"-r {remote_base}/requirements.txt",
                )

            server_cfg = self._config.text_embedding_server
            unit_content = _render_text_embedding_unit(
                port=endpoint.text_embedding_port,
                model=server_cfg.model_name,
                dimension=server_cfg.dimension,
                device=server_cfg.device,
            )
            unit_path = (
                f"{remote_home}/.config/systemd/user"
                f"/{_TEXT_EMBEDDING_UNIT_NAME}.service"
            )
            r = await conn.run(
                f"cat > {unit_path} << 'UNIT_EOF'\n{unit_content}UNIT_EOF"
            )
            if r.exit_status != 0:
                logger.error("Failed to write unit file: %s", r.stderr)
                conn.close()
                return False

            await conn.run("loginctl enable-linger $(whoami) 2>/dev/null || true")
            await conn.run("systemctl --user daemon-reload", check=True)
            await conn.run(
                f"systemctl --user enable {_TEXT_EMBEDDING_UNIT_NAME}", check=True
            )
            await conn.run(
                f"systemctl --user restart {_TEXT_EMBEDDING_UNIT_NAME}", check=True
            )

            import asyncio

            await asyncio.sleep(10)
            status_result = await conn.run(
                f"systemctl --user is-active {_TEXT_EMBEDDING_UNIT_NAME}"
            )
            is_active = status_result.stdout.strip() == "active"

            conn.close()

            if is_active:
                logger.info(
                    "Deployed text-embedding to %s (port %d, model: %s)",
                    endpoint.name, endpoint.text_embedding_port,
                    server_cfg.model_name,
                )
            else:
                logger.error(
                    "text-embedding deployed but not active on %s", endpoint.name
                )

            return is_active

        except Exception as e:
            logger.error("Deploy failed for %s: %s", endpoint.name, e)
            return False

    async def stop_text_embedding(self, endpoint: ServiceEndpoint) -> bool:
        """Stop text embedding service on a remote endpoint."""
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            await conn.run(f"systemctl --user stop {_TEXT_EMBEDDING_UNIT_NAME}")
            conn.close()

            logger.info("Stopped text-embedding on %s", endpoint.name)
            return True

        except Exception as e:
            logger.error("Stop failed for %s: %s", endpoint.name, e)
            return False

    async def status_text_embedding(
        self, endpoint: ServiceEndpoint
    ) -> ServiceHealth:
        """Check if text embedding is running on a remote endpoint."""
        from services.text_embedding.baseline.client import TextEmbeddingClient

        client = TextEmbeddingClient(endpoint.text_embedding)
        try:
            return await client.health()
        except Exception:
            return ServiceHealth(
                name=endpoint.name,
                status=ServiceStatus.STOPPED,
                endpoint=endpoint.text_embedding.base_url,
            )

    async def logs_text_embedding(
        self, endpoint: ServiceEndpoint, lines: int = 50
    ) -> str:
        """Fetch recent journal logs for text embedding on a remote endpoint."""
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            result = await conn.run(
                f"journalctl --user -u {_TEXT_EMBEDDING_UNIT_NAME} "
                f"-n {lines} --no-pager"
            )
            conn.close()
            return result.stdout

        except Exception as e:
            return f"Failed to fetch logs for {endpoint.name}: {e}"

    async def systemd_status_text_embedding(
        self, endpoint: ServiceEndpoint
    ) -> str:
        """Fetch systemctl status output for text embedding."""
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            result = await conn.run(
                f"systemctl --user status {_TEXT_EMBEDDING_UNIT_NAME} --no-pager"
            )
            conn.close()
            return result.stdout

        except Exception as e:
            return f"Failed to get status for {endpoint.name}: {e}"

    # --- Reranker methods ---

    async def deploy_reranker(self, endpoint: ServiceEndpoint) -> bool:
        """Deploy reranker service (SGLang + FastAPI) to a remote endpoint."""
        try:
            import asyncio

            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            home_result = await conn.run("echo $HOME", check=True)
            remote_home = home_result.stdout.strip()
            remote_base = f"{remote_home}/.reflection/services/reranker"

            await conn.run(
                f"mkdir -p {remote_base}/services/reranker/baseline "
                f"{remote_home}/.config/systemd/user",
                check=True,
            )

            project_root = Path(__file__).parent.parent
            async with conn.start_sftp_client() as sftp:
                for rel_path in _RERANKER_FILES:
                    local = project_root / rel_path
                    if not local.exists():
                        logger.warning("File not found: %s", local)
                        continue
                    remote = f"{remote_base}/{rel_path}"
                    await sftp.put(str(local), remote)

                req_local = project_root / _RERANKER_REQUIREMENTS
                if req_local.exists():
                    await sftp.put(str(req_local), f"{remote_base}/requirements.txt")

            # Create venv and install dependencies (including sglang)
            venv_path = f"{remote_base}/.venv"
            uv_bin = f"{remote_home}/.local/bin/uv"
            venv_check = await conn.run(f"test -f {venv_path}/bin/python")
            if venv_check.exit_status != 0:
                logger.info("Creating venv on %s ...", endpoint.name)
                await conn.run(f"{uv_bin} venv {venv_path}", check=True)
                logger.info("Installing dependencies on %s ...", endpoint.name)
                await conn.run(
                    f"{uv_bin} pip install --prerelease=allow"
                    f" -p {venv_path}/bin/python"
                    f" -r {remote_base}/requirements.txt ninja",
                    check=True,
                )
                logger.info("Dependencies installed on %s", endpoint.name)
            else:
                logger.info("Updating dependencies on %s ...", endpoint.name)
                await conn.run(
                    f"{uv_bin} pip install -q --prerelease=allow"
                    f" -p {venv_path}/bin/python"
                    f" -r {remote_base}/requirements.txt",
                )

            server_cfg = self._config.reranker_server

            # Write SGLang backend systemd unit
            backend_unit_content = _render_reranker_backend_unit(
                backend_port=server_cfg.vllm_port,
                model=server_cfg.model_name,
                device=server_cfg.device,
            )
            backend_unit_path = (
                f"{remote_home}/.config/systemd/user"
                f"/{_RERANKER_VLLM_UNIT_NAME}.service"
            )
            r = await conn.run(
                f"cat > {backend_unit_path} << 'UNIT_EOF'\n{backend_unit_content}UNIT_EOF"
            )
            if r.exit_status != 0:
                logger.error("Failed to write backend unit file: %s", r.stderr)
                conn.close()
                return False

            # Write FastAPI wrapper systemd unit
            unit_content = _render_reranker_unit(
                port=endpoint.reranker_port,
                vllm_port=server_cfg.vllm_port,
                model=server_cfg.model_name,
            )
            unit_path = (
                f"{remote_home}/.config/systemd/user"
                f"/{_RERANKER_UNIT_NAME}.service"
            )
            r = await conn.run(
                f"cat > {unit_path} << 'UNIT_EOF'\n{unit_content}UNIT_EOF"
            )
            if r.exit_status != 0:
                logger.error("Failed to write reranker unit file: %s", r.stderr)
                conn.close()
                return False

            await conn.run("loginctl enable-linger $(whoami) 2>/dev/null || true")
            await conn.run("systemctl --user daemon-reload", check=True)

            # Start SGLang backend first, then FastAPI wrapper
            await conn.run(
                f"systemctl --user enable {_RERANKER_VLLM_UNIT_NAME}", check=True
            )
            await conn.run(
                f"systemctl --user restart {_RERANKER_VLLM_UNIT_NAME}", check=True
            )
            # Wait for SGLang to load the model
            logger.info("Waiting for SGLang model to load on %s ...", endpoint.name)
            await asyncio.sleep(30)

            await conn.run(
                f"systemctl --user enable {_RERANKER_UNIT_NAME}", check=True
            )
            await conn.run(
                f"systemctl --user restart {_RERANKER_UNIT_NAME}", check=True
            )
            await asyncio.sleep(5)

            # Check both units are active
            vllm_status = await conn.run(
                f"systemctl --user is-active {_RERANKER_VLLM_UNIT_NAME}"
            )
            api_status = await conn.run(
                f"systemctl --user is-active {_RERANKER_UNIT_NAME}"
            )
            vllm_active = vllm_status.stdout.strip() == "active"
            api_active = api_status.stdout.strip() == "active"

            conn.close()

            if vllm_active and api_active:
                logger.info(
                    "Deployed reranker to %s (port %d, vllm port %d, model: %s)",
                    endpoint.name, endpoint.reranker_port,
                    server_cfg.vllm_port, server_cfg.model_name,
                )
            else:
                logger.error(
                    "Reranker deployed but not fully active on %s "
                    "(vllm=%s, api=%s)",
                    endpoint.name,
                    "active" if vllm_active else "inactive",
                    "active" if api_active else "inactive",
                )

            return vllm_active and api_active

        except Exception as e:
            logger.error("Deploy reranker failed for %s: %s", endpoint.name, e)
            return False

    async def stop_reranker(self, endpoint: ServiceEndpoint) -> bool:
        """Stop reranker services (FastAPI + vLLM) on a remote endpoint."""
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            # Stop FastAPI wrapper first, then vLLM
            await conn.run(f"systemctl --user stop {_RERANKER_UNIT_NAME}")
            await conn.run(f"systemctl --user stop {_RERANKER_VLLM_UNIT_NAME}")
            conn.close()

            logger.info("Stopped reranker on %s", endpoint.name)
            return True

        except Exception as e:
            logger.error("Stop reranker failed for %s: %s", endpoint.name, e)
            return False

    async def status_reranker(
        self, endpoint: ServiceEndpoint
    ) -> ServiceHealth:
        """Check if reranker is running on a remote endpoint."""
        from services.reranker.baseline.client import RerankerClient

        client = RerankerClient(endpoint.reranker)
        try:
            return await client.health()
        except Exception:
            return ServiceHealth(
                name=endpoint.name,
                status=ServiceStatus.STOPPED,
                endpoint=endpoint.reranker.base_url,
            )

    async def logs_reranker(
        self, endpoint: ServiceEndpoint, lines: int = 50
    ) -> str:
        """Fetch recent journal logs for reranker on a remote endpoint."""
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            # Fetch logs for both units
            api_result = await conn.run(
                f"journalctl --user -u {_RERANKER_UNIT_NAME} "
                f"-n {lines} --no-pager"
            )
            vllm_result = await conn.run(
                f"journalctl --user -u {_RERANKER_VLLM_UNIT_NAME} "
                f"-n {lines} --no-pager"
            )
            conn.close()
            return (
                f"--- {_RERANKER_UNIT_NAME} ---\n{api_result.stdout}\n"
                f"--- {_RERANKER_VLLM_UNIT_NAME} ---\n{vllm_result.stdout}"
            )

        except Exception as e:
            return f"Failed to fetch logs for {endpoint.name}: {e}"

    async def systemd_status_reranker(
        self, endpoint: ServiceEndpoint
    ) -> str:
        """Fetch systemctl status output for reranker."""
        try:
            import asyncssh

            conn = await asyncssh.connect(
                endpoint.host,
                port=endpoint.port,
                username=endpoint.user or None,
                known_hosts=None,
            )

            api_result = await conn.run(
                f"systemctl --user status {_RERANKER_UNIT_NAME} --no-pager"
            )
            vllm_result = await conn.run(
                f"systemctl --user status {_RERANKER_VLLM_UNIT_NAME} --no-pager"
            )
            conn.close()
            return (
                f"--- {_RERANKER_UNIT_NAME} ---\n{api_result.stdout}\n"
                f"--- {_RERANKER_VLLM_UNIT_NAME} ---\n{vllm_result.stdout}"
            )

        except Exception as e:
            return f"Failed to get status for {endpoint.name}: {e}"
