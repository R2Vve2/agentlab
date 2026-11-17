import asyncio

import pytest

mcp = pytest.importorskip("mcp", reason="optional dependency: pip install mcp")

from agentlab.mcp_server import build_server  # noqa: E402


def test_mcp_server_exposes_registry_tools():
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"lookup_order", "calculate", "get_policy"} <= names
    result = asyncio.run(server.call_tool("calculate", {"expression": "6*7"}))
    assert "42" in str(result)
