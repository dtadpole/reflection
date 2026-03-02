"""Configuration loading and validation."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class InsightFinderConfig(BaseModel):
    enabled: bool = True
    frequency: int = 5


class PipelineConfig(BaseModel):
    agents: list[str] = Field(
        default_factory=lambda: ["curator", "solver", "critic", "organizer"]
    )
    iterations: int = 1
    insight_finder: InsightFinderConfig = Field(default_factory=InsightFinderConfig)


class StorageConfig(BaseModel):
    data_root: str = "~/.reflection"
    env: str = ""  # Resolved at load time; see _default_env()
    lance_dir: str = "lance"

    @property
    def env_path(self) -> Path:
        """Root directory for this environment: <data_root>/<env>/"""
        return Path(self.data_root).expanduser() / self.env

    @property
    def lance_path(self) -> Path:
        return self.env_path / self.lance_dir

    @property
    def problems_path(self) -> Path:
        return self.env_path / "problems"

    @property
    def cards_path(self) -> Path:
        return self.env_path / "cards"

    @property
    def experiences_path(self) -> Path:
        return self.env_path / "experiences"

    @property
    def queues_path(self) -> Path:
        return self.env_path / "queues"

    @property
    def logs_path(self) -> Path:
        return self.env_path / "logs"

    def run_path(self, run_tag: str) -> Path:
        return self.env_path / run_tag

    def execution_log_path(self, run_tag: str, agent: str = "") -> Path:
        name = f"{agent}_output.log" if agent else "output.log"
        return self.run_path(run_tag) / name


def make_log_path(
    logs_dir: Path, agent_name: str, seq_id: int | None = None
) -> Path:
    """Build a log file path.

    Format: <agent_name>[_<seq_id>]_<datetime>.log
    seq_id is assigned by the orchestrator; omitted for standalone agents.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if seq_id is not None:
        name = f"{agent_name}_{seq_id}_{ts}.log"
    else:
        name = f"{agent_name}_{ts}.log"
    return logs_dir / name


class EmbedderConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 5



class KbEvalServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8456
    devices: list[str] = Field(default_factory=lambda: ["cuda:0"])
    max_critical_time: int = 120  # seconds, GPU lock timeout
    max_timeout: int = 600  # seconds, overall subprocess timeout
    code_type: str = "triton"  # triton | cuda | pytorch
    compile_pytorch: bool = False  # torch.compile reference model


class KbEvalClientConfig(BaseModel):
    base_url: str = "http://localhost:8456"
    timeout: int = 180  # HTTP read timeout (3 min max per eval)
    retry_count: int = 4
    retry_interval: float = 3.0


class TextEmbeddingServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 42982
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    dimension: int = 4096
    max_batch_size: int = 64
    max_seq_length: int = 8192
    device: str = "cuda:0"


class TextEmbeddingClientConfig(BaseModel):
    base_url: str = "http://localhost:42982"
    timeout: int = 60
    retry_count: int = 3
    retry_interval: float = 2.0


class RerankerServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 42983
    vllm_port: int = 42984
    model_name: str = "Qwen/Qwen3-32B"
    device: str = "cuda:0"


class RerankerClientConfig(BaseModel):
    base_url: str = "http://localhost:42983"
    timeout: int = 120
    retry_count: int = 3
    retry_interval: float = 2.0


class ServiceEndpoint(BaseModel):
    name: str
    host: str
    port: int = 22  # SSH port
    user: str = ""
    kb_eval_port: int = 8456  # kbEval server bind port on remote
    kb_eval: KbEvalClientConfig = Field(default_factory=KbEvalClientConfig)
    text_embedding_port: int = 42982
    text_embedding: TextEmbeddingClientConfig = Field(
        default_factory=TextEmbeddingClientConfig
    )
    reranker_port: int = 42983
    reranker: RerankerClientConfig = Field(default_factory=RerankerClientConfig)


class ServicesConfig(BaseModel):
    endpoints: list[ServiceEndpoint] = Field(default_factory=list)
    kb_eval_server: KbEvalServerConfig = Field(default_factory=KbEvalServerConfig)
    text_embedding_server: TextEmbeddingServerConfig = Field(
        default_factory=TextEmbeddingServerConfig
    )
    reranker_server: RerankerServerConfig = Field(
        default_factory=RerankerServerConfig
    )


class PortForward(BaseModel):
    local_port: int
    remote_port: int
    remote_host: str = "localhost"


class TunnelEndpoint(BaseModel):
    name: str
    host: str
    ssh_port: int = 22
    user: str = ""
    forwards: list[PortForward]


class TunnelsConfig(BaseModel):
    tunnels: list[TunnelEndpoint] = Field(default_factory=list)


class AgentSpec(BaseModel):
    name: str
    count: int = 1
    options: dict[str, str | int | bool] = Field(default_factory=dict)


class OrchestratorConfig(BaseModel):
    agents: list[AgentSpec] = Field(
        default_factory=lambda: [
            AgentSpec(name="curator", count=1, options={"n": 100, "levels": "level_1,level_2"}),
            AgentSpec(name="solver", count=2),
            AgentSpec(name="critic", count=1),
            AgentSpec(name="organizer", count=1, options={"interval": 300}),
            AgentSpec(name="insight_finder", count=1, options={"interval": 600}),
        ]
    )
    shutdown_timeout: int = 30
    status_interval: int = 60


class ReflectionConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    tunnels: TunnelsConfig = Field(default_factory=TunnelsConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)


def _default_env() -> str:
    """Return the default environment name: test_${USER}."""
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    return f"test_{user}"


def load_config(config_path: Optional[Path] = None) -> ReflectionConfig:
    """Load configuration from a TOML file, falling back to defaults.

    Also merges config/hosts.yaml (if present) into services.endpoints.
    """
    if config_path is None:
        config_path = Path("config/default.toml")

    if config_path.exists():
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        cfg = ReflectionConfig.model_validate(raw)
    else:
        cfg = ReflectionConfig()

    # Resolve default env if not explicitly set
    if not cfg.storage.env:
        cfg.storage.env = _default_env()

    # Merge hosts.yaml into services.endpoints
    hosts_path = config_path.parent / "hosts.yaml"
    if hosts_path.exists():
        import yaml

        with open(hosts_path) as f:
            hosts_raw = yaml.safe_load(f) or {}
        endpoints_raw = hosts_raw.get("endpoints", [])
        if endpoints_raw:
            cfg.services.endpoints = [
                ServiceEndpoint.model_validate(ep) for ep in endpoints_raw
            ]

    # Merge tunnels.yaml into tunnels config
    tunnels_path = config_path.parent / "tunnels.yaml"
    if tunnels_path.exists():
        import yaml

        with open(tunnels_path) as f:
            tunnels_raw = yaml.safe_load(f) or {}
        cfg.tunnels = TunnelsConfig.model_validate(tunnels_raw)

    return cfg
