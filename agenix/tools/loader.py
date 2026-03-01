"""Load tool definitions from tools/<name>/<variant>/ folders."""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

# Default tools directory relative to project root
_DEFAULT_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools"
_DEFAULT_VARIANT = "base"


def parse_tool_md(text: str) -> dict[str, str]:
    """Parse a tool.md file into sections.

    Expected sections: Description, Input Schema, Output Schema, Examples.
    Returns a dict keyed by lowercase section name (spaces replaced with _).
    """
    sections: dict[str, str] = {}
    current_section: Optional[str] = None
    current_lines: list[str] = []
    tool_name = ""

    for line in text.splitlines():
        # Top-level heading = tool name
        if re.match(r"^# ", line) and not tool_name:
            tool_name = line.lstrip("# ").strip()
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

    sections["name"] = tool_name
    return sections


@dataclass
class LoadedTool:
    """A loaded tool definition with its metadata and factory function."""

    name: str
    variant: str
    description: str
    input_schema: str
    output_schema: str
    examples: str
    config: dict[str, Any]
    create_fn: Callable[..., Any]


def load_tool(
    tool_name: str,
    variant: str = _DEFAULT_VARIANT,
    tools_dir: Optional[Path] = None,
) -> LoadedTool:
    """Load a complete tool definition from its folder.

    Path: tools/<tool_name>/<variant>/

    Reads:
    - tool.md -> description, input_schema, output_schema, examples
    - config.yaml -> tool-specific config dict
    - logic.py -> create_tool() factory callable
    """
    if tools_dir is None:
        tools_dir = _DEFAULT_TOOLS_DIR

    variant_dir = tools_dir / tool_name / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"Tool variant directory not found: {variant_dir}")

    # Parse tool.md
    tool_md_path = variant_dir / "tool.md"
    if not tool_md_path.exists():
        raise FileNotFoundError(f"tool.md not found in {variant_dir}")

    sections = parse_tool_md(tool_md_path.read_text())

    # Load config.yaml
    config_path = variant_dir / "config.yaml"
    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Import logic.py and extract create_tool
    logic_path = variant_dir / "logic.py"
    if not logic_path.exists():
        raise FileNotFoundError(f"logic.py not found in {variant_dir}")

    module_name = f"tools.{tool_name}.{variant}.logic"
    spec = importlib.util.spec_from_file_location(module_name, logic_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {logic_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    create_fn = getattr(module, "create_tool", None)
    if create_fn is None:
        raise AttributeError(f"logic.py in {variant_dir} must export a 'create_tool' function")

    return LoadedTool(
        name=config.get("name", tool_name),
        variant=config.get("variant", variant),
        description=sections.get("description", ""),
        input_schema=sections.get("input_schema", ""),
        output_schema=sections.get("output_schema", ""),
        examples=sections.get("examples", ""),
        config=config,
        create_fn=create_fn,
    )


def list_tools(tools_dir: Optional[Path] = None) -> list[str]:
    """List all available tool names."""
    if tools_dir is None:
        tools_dir = _DEFAULT_TOOLS_DIR

    if not tools_dir.is_dir():
        return []

    return sorted(
        d.name
        for d in tools_dir.iterdir()
        if d.is_dir() and _has_any_variant(d)
    )


def list_variants(
    tool_name: str, tools_dir: Optional[Path] = None
) -> list[str]:
    """List all available variants for a tool."""
    if tools_dir is None:
        tools_dir = _DEFAULT_TOOLS_DIR

    tool_dir = tools_dir / tool_name
    if not tool_dir.is_dir():
        return []

    return sorted(
        d.name
        for d in tool_dir.iterdir()
        if d.is_dir() and (d / "tool.md").exists()
    )


def _has_any_variant(tool_dir: Path) -> bool:
    """Check if a tool directory has at least one variant with tool.md."""
    if not tool_dir.is_dir():
        return False
    return any(
        (d / "tool.md").exists()
        for d in tool_dir.iterdir()
        if d.is_dir()
    )
