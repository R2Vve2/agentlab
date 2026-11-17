import pytest

from agentlab.store_tools import build_store_registry
from agentlab.tools import ToolRegistry


@pytest.fixture(scope="module")
def registry():
    return build_store_registry()


def test_schema_generated_from_type_hints():
    reg = ToolRegistry()

    @reg.tool("Add numbers", params={"a": "first"})
    def add(a: int, b: float = 1.5) -> float:
        return a + b

    api = reg.to_api()[0]
    assert api["name"] == "add" and api["strict"] is True
    assert api["input_schema"]["properties"]["a"] == {"type": "integer", "description": "first"}
    assert api["input_schema"]["properties"]["b"] == {"type": "number", "default": 1.5}
    assert api["input_schema"]["required"] == ["a"]
    assert reg.call("add", {"a": 2}).output == "3.5"


def test_invalid_input_becomes_error_result():
    reg = ToolRegistry()

    @reg.tool("echo")
    def echo(text: str) -> str:
        return text

    r = reg.call("echo", {"wrong": 1})
    assert r.is_error and "Invalid input" in r.output
    r = reg.call("nope", {})
    assert r.is_error and "Unknown tool" in r.output


def test_exceptions_are_captured(registry):
    r = registry.call("lookup_order", {"order_id": "ORD-0000"})
    assert r.is_error and "not found" in r.output


def test_unsupported_annotation_rejected():
    reg = ToolRegistry()
    with pytest.raises(TypeError):

        @reg.tool("bad")
        def bad(x: list) -> str:
            return "x"


def test_store_tools(registry):
    assert set(registry.names()) == {"lookup_order", "orders_for_customer", "search_products", "get_policy", "calculate", "days_between", "today"}
    order = registry.call("lookup_order", {"order_id": "ord-1002"})
    assert not order.is_error and "DHL-77812390" in order.output
    assert registry.call("calculate", {"expression": "2*89 + 39 + 9.99"}).output == "226.99"
    assert registry.call("calculate", {"expression": "__import__('os')"}).is_error
    assert registry.call("days_between", {"start": "2026-11-05", "end": "2026-11-17"}).output == "12"
    assert registry.call("today", {}).output == "2026-11-17"
    hits = registry.call("search_products", {"query": "monitor"}).output
    assert "MN-240" in hits and "HP-220" not in hits
    assert registry.call("get_policy", {"topic": "exchanges"}).is_error
    assert "ORD-1001" in registry.call("orders_for_customer", {"email": "SARA@example.com"}).output
