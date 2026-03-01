"""Load agent definitions from agents/<name>/<variant>/ folders."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import yaml

from agenix.storage.models import AgentConfig, LoadedAgent

# Default agents directory relative to project root
_DEFAULT_AGENTS_DIR = Path(__file__).parent.parent / "agents"
_DEFAULT_VARIANT = "base"


def parse_agent_md(text: str) -> dict[str, str]:
    """Parse an agent.md file into sections.

    Expected sections: Description, System Prompt, Input Format, Output Format, Examples.
    Returns a dict keyed by lowercase section name.
    """
    sections: dict[str, str] = {}
    current_section: Optional[str] = None
    current_lines: list[str] = []
    agent_name = ""

    for line in text.splitlines():
        # Top-level heading = agent name
        if re.match(r"^# ", line) and not agent_name:
            agent_name = line.lstrip("# ").strip()
            continue

        # Section heading
        h2_match = re.match(r"^## (.+)$", line)
        if h2_match:
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = h2_match.group(1).strip().lower().replace(" ", "_")
            current_lines = []
            continue

        current_lines.append(line)

    # Flush last section
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    sections["name"] = agent_name
    return sections


def load_agent_config(config_path: Path) -> AgentConfig:
    """Load agent configuration from a YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return AgentConfig.model_validate(raw)


def load_agent_logic(logic_path: Path, agent_name: str, variant: str) -> ModuleType:
    """Dynamically import an agent's logic.py module."""
    module_name = f"agents.{agent_name}.{variant}.logic"
    spec = importlib.util.spec_from_file_location(module_name, logic_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {logic_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_agent(
    agent_name: str,
    variant: str = _DEFAULT_VARIANT,
    agents_dir: Optional[Path] = None,
) -> LoadedAgent:
    """Load a complete agent definition from its folder.

    Path: agents/<agent_name>/<variant>/

    Reads:
    - agent.md → description, system_prompt, input_format, output_format, examples
    - config.yaml → model, temperature, max_turns, tools, custom_tools
    - logic.py → optional Python module path (loaded on demand)
    """
    if agents_dir is None:
        agents_dir = _DEFAULT_AGENTS_DIR

    variant_dir = agents_dir / agent_name / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"Agent variant directory not found: {variant_dir}")

    # Parse agent.md
    agent_md_path = variant_dir / "agent.md"
    if not agent_md_path.exists():
        raise FileNotFoundError(f"agent.md not found in {variant_dir}")

    sections = parse_agent_md(agent_md_path.read_text())

    # Load config.yaml
    config_path = variant_dir / "config.yaml"
    config = load_agent_config(config_path) if config_path.exists() else AgentConfig()

    # Check for logic.py
    logic_path = variant_dir / "logic.py"
    logic_module_path = str(logic_path) if logic_path.exists() else None

    return LoadedAgent(
        name=sections.get("name", agent_name),
        description=sections.get("description", ""),
        system_prompt=sections.get("system_prompt", ""),
        input_format=sections.get("input_format", ""),
        output_format=sections.get("output_format", ""),
        examples=sections.get("examples", ""),
        config=config,
        logic_module_path=logic_module_path,
        variant=variant,
    )


def list_agents(agents_dir: Optional[Path] = None) -> list[str]:
    """List all available agent names."""
    if agents_dir is None:
        agents_dir = _DEFAULT_AGENTS_DIR

    if not agents_dir.is_dir():
        return []

    return sorted(
        d.name
        for d in agents_dir.iterdir()
        if d.is_dir() and _has_any_variant(d)
    )


def list_variants(
    agent_name: str, agents_dir: Optional[Path] = None
) -> list[str]:
    """List all available variants for an agent."""
    if agents_dir is None:
        agents_dir = _DEFAULT_AGENTS_DIR

    agent_dir = agents_dir / agent_name
    if not agent_dir.is_dir():
        return []

    return sorted(
        d.name
        for d in agent_dir.iterdir()
        if d.is_dir() and (d / "agent.md").exists()
    )


def _has_any_variant(agent_dir: Path) -> bool:
    """Check if an agent directory has at least one variant with agent.md."""
    if not agent_dir.is_dir():
        return False
    return any(
        (d / "agent.md").exists()
        for d in agent_dir.iterdir()
        if d.is_dir()
    )
