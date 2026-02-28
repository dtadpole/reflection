"""Load agent definitions from agents/<name>/ folders."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from agenix.storage.models import AgentConfig, LoadedAgent

# Default agents directory relative to project root
_DEFAULT_AGENTS_DIR = Path(__file__).parent.parent / "agents"


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
    """Load agent configuration from a TOML file."""
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    return AgentConfig.model_validate(raw)


def load_agent_logic(logic_path: Path, agent_name: str) -> ModuleType:
    """Dynamically import an agent's logic.py module."""
    module_name = f"agents.{agent_name}.logic"
    spec = importlib.util.spec_from_file_location(module_name, logic_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {logic_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_agent(agent_name: str, agents_dir: Optional[Path] = None) -> LoadedAgent:
    """Load a complete agent definition from its folder.

    Reads:
    - agent.md → description, system_prompt, input_format, output_format, examples
    - config.toml → model, temperature, max_turns, tools, custom_tools
    - logic.py → optional Python module path (loaded on demand)
    """
    if agents_dir is None:
        agents_dir = _DEFAULT_AGENTS_DIR

    agent_dir = agents_dir / agent_name
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    # Parse agent.md
    agent_md_path = agent_dir / "agent.md"
    if not agent_md_path.exists():
        raise FileNotFoundError(f"agent.md not found in {agent_dir}")

    sections = parse_agent_md(agent_md_path.read_text())

    # Load config.toml
    config_path = agent_dir / "config.toml"
    config = load_agent_config(config_path) if config_path.exists() else AgentConfig()

    # Check for logic.py
    logic_path = agent_dir / "logic.py"
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
        if d.is_dir() and (d / "agent.md").exists()
    )
