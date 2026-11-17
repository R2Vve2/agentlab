"""Structured traces for every agent run: each model call and tool call becomes a span with
timing, token usage and a cost estimate. Traces are exported as JSONL so they can be
inspected, diffed between prompt versions, or loaded into an observability tool."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# USD per million tokens (input, output). Used only for estimates in traces and eval reports.
PRICING = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "scripted": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return round((input_tokens * inp + output_tokens * out) / 1_000_000, 6)


@dataclass
class Span:
    kind: str  # "llm" | "tool"
    name: str
    input: Any
    output: Any
    duration_ms: float
    is_error: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Trace:
    task: str
    model: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    spans: list[Span] = field(default_factory=list)
    final_answer: str = ""
    status: str = "running"  # running | completed | max_steps | refused | error
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)  # full transcript, not exported

    @property
    def steps(self) -> int:
        return sum(1 for s in self.spans if s.kind == "llm")

    @property
    def tool_calls(self) -> list[str]:
        return [s.name for s in self.spans if s.kind == "tool"]

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    @property
    def duration_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.spans), 1)

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("messages", None)
        return {
            **data,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
        }


class TraceStore:
    def __init__(self, path: str | Path = "traces/traces.jsonl"):
        self.path = Path(path)

    def append(self, trace: Trace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n")

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
