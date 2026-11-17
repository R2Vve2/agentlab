"""Tools for the demo scenario: a support agent for a small online electronics store.

The "backend" is a few JSON files under ``data/`` so the agent is fully runnable offline.
Each tool is deliberately small and returns structured text; the model composes them.
"""

from __future__ import annotations

import ast
import json
import operator
from datetime import date, datetime
from pathlib import Path

from agentlab.tools import ToolRegistry

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TODAY = date(2026, 11, 17)  # frozen so evals are reproducible


def _load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


# --- safe arithmetic ---------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


def build_store_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool("Look up an order by its id (format ORD-####). Returns status, items, totals and dates.",
                   params={"order_id": "The order id, e.g. ORD-1001"})
    def lookup_order(order_id: str) -> dict:
        orders = _load("orders.json")
        key = order_id.strip().upper()
        if key not in orders:
            raise KeyError(f"order {key} not found")
        return orders[key]

    @registry.tool("List all orders for a customer email address.", params={"email": "Customer email"})
    def orders_for_customer(email: str) -> list[dict]:
        orders = _load("orders.json")
        found = [{"order_id": oid, "status": o["status"], "total": o["total"], "ordered_on": o["ordered_on"]}
                 for oid, o in orders.items() if o["customer_email"].lower() == email.strip().lower()]
        return found

    @registry.tool("Search the product catalogue by keyword. Returns name, sku, price, stock and warranty months.",
                   params={"query": "Free-text search, e.g. 'headphones'", "max_results": "Maximum number of results"})
    def search_products(query: str, max_results: int = 5) -> list[dict]:
        products = _load("products.json")
        q = query.lower().split()
        scored = [(sum(t in (p["name"] + " " + p["category"] + " " + p["sku"]).lower() for t in q), p) for p in products]
        hits = [p for s, p in sorted(scored, key=lambda sp: -sp[0]) if s > 0]
        return hits[:max_results]

    @registry.tool("Get the store policy text for a topic: 'returns', 'refunds', 'shipping' or 'warranty'.",
                   params={"topic": "One of returns, refunds, shipping, warranty"})
    def get_policy(topic: str) -> str:
        policies = _load("policies.json")
        key = topic.strip().lower()
        if key not in policies:
            raise KeyError(f"no policy for {key!r}; available: {', '.join(policies)}")
        return policies[key]

    @registry.tool("Evaluate an arithmetic expression exactly, e.g. '2 * (349.00 - 50)'.", params={"expression": "Arithmetic expression"})
    def calculate(expression: str) -> str:
        value = _eval(ast.parse(expression, mode="eval").body)
        return f"{value:.2f}" if isinstance(value, float) and not value.is_integer() else str(int(value))

    @registry.tool("Number of days between two ISO dates (YYYY-MM-DD); positive when the second date is later.",
                   params={"start": "Start date", "end": "End date"})
    def days_between(start: str, end: str) -> int:
        return (datetime.fromisoformat(end).date() - datetime.fromisoformat(start).date()).days

    @registry.tool("Today's date in ISO format.")
    def today() -> str:
        return TODAY.isoformat()

    return registry
