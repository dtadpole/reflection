"""Configuration loading and validation."""

from __future__ import annotations

import sys
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
    env: str = "prod"
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
    def queues_path(self) -> Path:
        return self.env_path / "queues"

    def run_path(self, run_tag: str) -> Path:
        return self.env_path / run_tag


class EmbedderConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 5


class CodeExecutorConfig(BaseModel):
    timeout_seconds: int = 30
    max_output_bytes: int = 65536


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
    timeout: int = 300  # HTTP read timeout
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
    model_name: str = "Qwen/Qwen3.5-27B"
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


class ReflectionConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    code_executor: CodeExecutorConfig = Field(default_factory=CodeExecutorConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    tunnels: TunnelsConfig = Field(default_factory=TunnelsConfig)


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
