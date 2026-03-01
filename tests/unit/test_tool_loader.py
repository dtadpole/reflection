"""Tests for tool loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenix.tools.loader import (
    LoadedTool,
    list_tools,
    list_variants,
    load_tool,
    parse_tool_md,
)

# --- Fixtures ---


@pytest.fixture
def sample_tools_dir(tmp_path: Path) -> Path:
    """Create a sample tools directory: tools/my_tool/base/"""
    variant_dir = tmp_path / "tools" / "my_tool" / "base"
    variant_dir.mkdir(parents=True)

    (variant_dir / "tool.md").write_text(
        """# My Tool

## Description
A test tool for unit tests.

## Input Schema
- query (str, required): Search query

## Output Schema
- results (list): Matching items

## Examples
Input: {"query": "hello"}
Output: {"results": ["world"]}
"""
    )

    (variant_dir / "config.yaml").write_text(
        "name: my_tool\nvariant: base\n"
    )

    (variant_dir / "logic.py").write_text(
        """def create_tool(**kwargs):
    return {"name": "my_tool", "kwargs": kwargs}
"""
    )

    return tmp_path / "tools"


@pytest.fixture
def multi_variant_tools_dir(tmp_path: Path) -> Path:
    """Create a tools dir with multiple tools and variants."""
    # tool_a/base
    base_dir = tmp_path / "tools" / "tool_a" / "base"
    base_dir.mkdir(parents=True)
    (base_dir / "tool.md").write_text("# Tool A\n\n## Description\nBase variant.")
    (base_dir / "config.yaml").write_text("name: tool_a\nvariant: base\n")
    (base_dir / "logic.py").write_text("def create_tool(**kwargs): return 'a_base'\n")

    # tool_a/v2
    v2_dir = tmp_path / "tools" / "tool_a" / "v2"
    v2_dir.mkdir(parents=True)
    (v2_dir / "tool.md").write_text("# Tool A V2\n\n## Description\nV2 variant.")
    (v2_dir / "config.yaml").write_text("name: tool_a\nvariant: v2\n")
    (v2_dir / "logic.py").write_text("def create_tool(**kwargs): return 'a_v2'\n")

    # tool_b/base
    b_dir = tmp_path / "tools" / "tool_b" / "base"
    b_dir.mkdir(parents=True)
    (b_dir / "tool.md").write_text("# Tool B\n\n## Description\nOnly variant.")
    (b_dir / "config.yaml").write_text("name: tool_b\nvariant: base\n")
    (b_dir / "logic.py").write_text("def create_tool(**kwargs): return 'b_base'\n")

    return tmp_path / "tools"


# --- parse_tool_md ---


class TestParseToolMd:
    def test_parse_full(self):
        text = """# My Tool

## Description
A helpful tool.

## Input Schema
- query (str): Search query

## Output Schema
- results (list): Items

## Examples
Input: {"query": "hello"}
Output: {"results": []}
"""
        sections = parse_tool_md(text)
        assert sections["name"] == "My Tool"
        assert sections["description"] == "A helpful tool."
        assert "query" in sections["input_schema"]
        assert "results" in sections["output_schema"]
        assert "hello" in sections["examples"]

    def test_parse_minimal(self):
        text = """# Tool

## Description
Minimal.
"""
        sections = parse_tool_md(text)
        assert sections["name"] == "Tool"
        assert sections["description"] == "Minimal."

    def test_parse_no_examples(self):
        text = """# Tool

## Description
No examples.

## Input Schema
- param (str)

## Output Schema
- result (str)
"""
        sections = parse_tool_md(text)
        assert "examples" not in sections
        assert sections["input_schema"] == "- param (str)"


# --- load_tool ---


class TestLoadTool:
    def test_load_full_tool(self, sample_tools_dir: Path):
        loaded = load_tool("my_tool", tools_dir=sample_tools_dir)
        assert isinstance(loaded, LoadedTool)
        assert loaded.name == "my_tool"
        assert loaded.variant == "base"
        assert "test tool" in loaded.description.lower()
        assert "query" in loaded.input_schema
        assert "results" in loaded.output_schema
        assert "hello" in loaded.examples
        assert loaded.config["name"] == "my_tool"
        assert loaded.create_fn is not None

    def test_create_fn_callable(self, sample_tools_dir: Path):
        loaded = load_tool("my_tool", tools_dir=sample_tools_dir)
        result = loaded.create_fn(foo="bar")
        assert result == {"name": "my_tool", "kwargs": {"foo": "bar"}}

    def test_load_specific_variant(self, multi_variant_tools_dir: Path):
        loaded = load_tool("tool_a", variant="v2", tools_dir=multi_variant_tools_dir)
        assert loaded.variant == "v2"
        assert loaded.create_fn() == "a_v2"

    def test_load_default_variant(self, multi_variant_tools_dir: Path):
        loaded = load_tool("tool_a", tools_dir=multi_variant_tools_dir)
        assert loaded.variant == "base"
        assert loaded.create_fn() == "a_base"

    def test_load_nonexistent_tool(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_tool("nonexistent", tools_dir=tools_dir)

    def test_load_nonexistent_variant(self, sample_tools_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_tool("my_tool", variant="v2", tools_dir=sample_tools_dir)

    def test_load_missing_tool_md(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        variant_dir = tools_dir / "bad_tool" / "base"
        variant_dir.mkdir(parents=True)
        (variant_dir / "config.yaml").write_text("name: bad_tool\n")
        (variant_dir / "logic.py").write_text("def create_tool(): pass\n")
        with pytest.raises(FileNotFoundError, match="tool.md"):
            load_tool("bad_tool", tools_dir=tools_dir)

    def test_load_missing_logic_py(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        variant_dir = tools_dir / "no_logic" / "base"
        variant_dir.mkdir(parents=True)
        (variant_dir / "tool.md").write_text("# No Logic\n\n## Description\nMissing logic.")
        (variant_dir / "config.yaml").write_text("name: no_logic\n")
        with pytest.raises(FileNotFoundError, match="logic.py"):
            load_tool("no_logic", tools_dir=tools_dir)

    def test_load_missing_create_tool(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        variant_dir = tools_dir / "bad_logic" / "base"
        variant_dir.mkdir(parents=True)
        (variant_dir / "tool.md").write_text("# Bad\n\n## Description\nBad logic.")
        (variant_dir / "config.yaml").write_text("name: bad_logic\n")
        (variant_dir / "logic.py").write_text("def wrong_name(): pass\n")
        with pytest.raises(AttributeError, match="create_tool"):
            load_tool("bad_logic", tools_dir=tools_dir)


# --- list_tools ---


class TestListTools:
    def test_list_tools(self, multi_variant_tools_dir: Path):
        tools = list_tools(tools_dir=multi_variant_tools_dir)
        assert tools == ["tool_a", "tool_b"]

    def test_list_empty(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        assert list_tools(tools_dir=tools_dir) == []

    def test_list_ignores_non_tool_dirs(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        # Dir without any variant containing tool.md
        (tools_dir / "not_a_tool").mkdir(parents=True)
        # Dir with a variant containing tool.md
        variant_dir = tools_dir / "real_tool" / "base"
        variant_dir.mkdir(parents=True)
        (variant_dir / "tool.md").write_text("# Real\n\n## Description\nReal.")
        tools = list_tools(tools_dir=tools_dir)
        assert tools == ["real_tool"]

    def test_list_nonexistent_dir(self, tmp_path: Path):
        assert list_tools(tools_dir=tmp_path / "nope") == []


# --- list_variants ---


class TestListVariants:
    def test_list_single_variant(self, sample_tools_dir: Path):
        variants = list_variants("my_tool", tools_dir=sample_tools_dir)
        assert variants == ["base"]

    def test_list_multiple_variants(self, multi_variant_tools_dir: Path):
        variants = list_variants("tool_a", tools_dir=multi_variant_tools_dir)
        assert variants == ["base", "v2"]

    def test_list_variants_nonexistent(self, tmp_path: Path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        assert list_variants("nope", tools_dir=tools_dir) == []
