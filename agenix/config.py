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


class ReflectionConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    code_executor: CodeExecutorConfig = Field(default_factory=CodeExecutorConfig)


def load_config(config_path: Optional[Path] = None) -> ReflectionConfig:
    """Load configuration from a TOML file, falling back to defaults."""
    if config_path is None:
        config_path = Path("config/default.toml")

    if config_path.exists():
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        return ReflectionConfig.model_validate(raw)

    return ReflectionConfig()
