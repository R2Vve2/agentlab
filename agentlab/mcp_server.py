"""Expose the store tools as an MCP (Model Context Protocol) server over stdio.

    pip install "mcp<2"
    python -m agentlab.mcp_server

Any MCP-capable client (Claude Desktop, Claude Code, an IDE) can then call the same tools the
agent uses. The registry's JSON schemas are reused, so a tool is defined once.
"""

from __future__ import annotations

from typing import Any

from agentlab.store_tools import build_store_registry
from agentlab.tools import ToolRegistry


def build_server(registry: ToolRegistry | None = None) -> Any:
    from mcp.server.fastmcp import FastMCP  # optional dependency

    registry = registry or build_store_registry()
    server = FastMCP("agentlab-store")
    for name in registry.names():
        spec = registry.get(name)
        # FastMCP derives the schema from the function signature and docstring, which is exactly
        # what the registry built its own schema from, so the two stay in sync.
        spec.func.__doc__ = spec.description
        server.tool(name=name, description=spec.description)(spec.func)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
