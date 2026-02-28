"""Tool registry and MCP server creation for agenix agents."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server


class ToolRegistry:
    """Registry for custom tools that can be exposed as MCP servers."""

    def __init__(self) -> None:
        self._tools: dict[str, SdkMcpTool[Any]] = {}

    def register(self, tool: SdkMcpTool[Any]) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> SdkMcpTool[Any]:
        """Get a registered tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name]

    def get_tools(self, names: list[str]) -> list[SdkMcpTool[Any]]:
        """Get multiple tools by name."""
        return [self.get(name) for name in names]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())

    def create_mcp_server(
        self,
        server_name: str,
        tool_names: list[str],
        version: str = "1.0.0",
    ) -> Any:
        """Create an in-process MCP server with the specified tools.

        Returns an McpSdkServerConfig suitable for ClaudeAgentOptions.mcp_servers.
        """
        tools = self.get_tools(tool_names)
        return create_sdk_mcp_server(
            name=server_name,
            version=version,
            tools=tools,
        )


# Global registry instance
registry = ToolRegistry()
