"""Evaluation harness.

A task file lists scenarios with deterministic graders:

* ``expected_contains``  - every substring must appear in the final answer (case-insensitive)
* ``expected_number``    - a number in the answer must be within ``tolerance`` of this value
* ``must_call``          - these tools must have been called at least once
* ``must_not_call``      - these tools must never be called (e.g. no order lookup for a catalogue question)
* ``max_steps``          - the run must finish within this many model calls

Two run modes:

* ``scripted`` replays the ``script`` stored with each task through ``ScriptedLLM``. It exercises
  the agent loop, tools and graders deterministically (CI), and documents the ideal trajectory.
* ``claude`` runs the real model and measures it against the same graders.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentlab.agent import Agent
from agentlab.llm import LLM, ScriptedLLM
from agentlab.tools import ToolRegistry
from agentlab.tracing import Trace, TraceStore

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    failures: list[str]
    steps: int
    tool_calls: list[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: float
    final_answer: str
    status: str


def grade(task: dict[str, Any], trace: Trace) -> list[str]:
    failures: list[str] = []
    answer = trace.final_answer or ""
    low = answer.lower()
    if trace.status != "completed":
        failures.append(f"status={trace.status}")
    for s in task.get("expected_contains", []):
        if s.lower() not in low:
            failures.append(f"missing text {s!r}")
    if "expected_number" in task:
        target, tol = float(task["expected_number"]), float(task.get("tolerance", 0.005))
        nums = [float(n.replace(",", "")) for n in _NUMBER.findall(answer)]
        if not any(abs(n - target) <= tol for n in nums):
            failures.append(f"number {target} not found (saw {nums[:6]})")
    called = set(trace.tool_calls)
    for t in task.get("must_call", []):
        if t not in called:
            failures.append(f"tool {t} not called")
    for t in task.get("must_not_call", []):
        if t in called:
            failures.append(f"tool {t} must not be called")
    if "max_steps" in task and trace.steps > task["max_steps"]:
        failures.append(f"{trace.steps} steps > max {task['max_steps']}")
    return failures


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["tasks"]


def _scripted_llm(task: dict[str, Any]) -> ScriptedLLM:
    turns: list[Any] = []
    for turn in task["script"]:
        if isinstance(turn, str):
            turns.append(turn)
        else:
            turns.append([ScriptedLLM.tool_call(c["tool"], **c.get("input", {})) for c in turn])
    return ScriptedLLM(turns)


def run_eval(
    tasks: list[dict[str, Any]],
    registry: ToolRegistry,
    llm: LLM | None = None,
    system: str | None = None,
    max_steps: int = 10,
    trace_store: TraceStore | None = None,
    only: set[str] | None = None,
) -> dict[str, Any]:
    results: list[TaskResult] = []
    for task in tasks:
        if only and task["id"] not in only:
            continue
        model = llm or _scripted_llm(task)
        kwargs = {"system": system} if system else {}
        agent = Agent(model, registry, max_steps=max_steps, trace_store=trace_store, **kwargs)
        try:
            trace = agent.run(task["prompt"])
        except Exception as exc:  # noqa: BLE001 - one failing task must not abort the suite
            trace = Trace(task=task["prompt"], model=getattr(model, "model", "?"), status="error", error=str(exc))
        failures = grade(task, trace)
        results.append(TaskResult(task["id"], not failures, failures, trace.steps, trace.tool_calls, trace.input_tokens,
                                  trace.output_tokens, trace.cost_usd, trace.duration_ms, trace.final_answer, trace.status))
    n = len(results) or 1
    summary = {
        "mode": (llm.model if llm else "scripted"),
        "tasks": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / n, 3),
        "avg_steps": round(statistics.mean(r.steps for r in results), 2) if results else 0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "total_input_tokens": sum(r.input_tokens for r in results),
        "total_output_tokens": sum(r.output_tokens for r in results),
    }
    return {"summary": summary, "results": [asdict(r) for r in results]}


def format_report(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [f"AgentLab eval  mode={s['mode']}  pass {s['passed']}/{s['tasks']} ({s['pass_rate']:.0%})  avg steps {s['avg_steps']}  est. cost ${s['total_cost_usd']}",
             "-" * 96]
    for r in report["results"]:
        flag = "PASS" if r["passed"] else "FAIL"
        tools = ",".join(r["tool_calls"]) or "-"
        lines.append(f"{flag}  {r['task_id']:<28} steps={r['steps']}  tools={tools}")
        for f in r["failures"]:
            lines.append(f"        - {f}")
    return "\n".join(lines)
