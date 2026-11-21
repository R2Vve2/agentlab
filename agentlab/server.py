"""Web server for AgentLab.

    uvicorn agentlab.server:app --reload        # http://localhost:8000

Two modes, both on one page:

* **Replay** (always available, free): any of the graded eval tasks is run through the real agent
  loop and real tools with ``ScriptedLLM`` playing the model, and the full trace is shown with its
  grade. No API key, no cost.
* **Live** (only when ``ANTHROPIC_API_KEY`` is set): visitors ask their own questions and Claude
  drives the agent. Guarded by a per-IP rate limit and a daily run cap so a public deployment
  cannot run up an unbounded bill.

A tool playground calls any registered tool directly, which shows the typed tool layer on its own.

Settings (environment variables): ``AGENTLAB_MODEL`` (default ``claude-opus-5-5``),
``AGENTLAB_EFFORT`` (optional: low/medium/high/xhigh/max), ``AGENTLAB_MAX_STEPS`` (default 8),
``AGENTLAB_RATE_LIMIT_PER_MINUTE`` (default 5 live runs per IP), ``AGENTLAB_DAILY_RUN_LIMIT``
(default 100 live runs per UTC day).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agentlab import __version__
from agentlab.agent import Agent
from agentlab.evals import _scripted_llm, grade, load_tasks
from agentlab.llm import LLM, build_llm
from agentlab.store_tools import build_store_registry
from agentlab.tracing import Trace

log = logging.getLogger("agentlab.server")
ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "ui" / "index.html"
TASKS_PATH = ROOT / "evals" / "tasks.json"


@dataclass
class ServerSettings:
    live: bool = False
    model: str = "claude-opus-5-5"
    effort: str | None = None
    max_steps: int = 8
    rate_limit_per_minute: int = 5
    daily_run_limit: int = 100

    @classmethod
    def from_env(cls) -> ServerSettings:
        return cls(
            live=bool(os.environ.get("ANTHROPIC_API_KEY")),
            model=os.environ.get("AGENTLAB_MODEL", "claude-opus-5-5"),
            effort=os.environ.get("AGENTLAB_EFFORT") or None,
            max_steps=int(os.environ.get("AGENTLAB_MAX_STEPS", "8")),
            rate_limit_per_minute=int(os.environ.get("AGENTLAB_RATE_LIMIT_PER_MINUTE", "5")),
            daily_run_limit=int(os.environ.get("AGENTLAB_DAILY_RUN_LIMIT", "100")),
        )


class RateLimiter:
    """Sliding one-minute window per key. In-memory: fine for a single-instance demo."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit, self.window = limit, window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


class DailyBudget:
    """Counts live runs per UTC day and refuses once the cap is reached."""

    def __init__(self, limit: int, today: Callable[[], str] | None = None):
        self.limit = limit
        self._today = today or (lambda: datetime.now(UTC).date().isoformat())
        self._day, self.used = self._today(), 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day, self.used = day, 0
            if self.limit > 0 and self.used >= self.limit:
                return False
            self.used += 1
            return True

    @property
    def remaining(self) -> int | None:
        return None if self.limit <= 0 else max(self.limit - self.used, 0)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def trace_to_json(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "status": trace.status,
        "final_answer": trace.final_answer,
        "steps": trace.steps,
        "tool_calls": trace.tool_calls,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "cost_usd": trace.cost_usd,
        "duration_ms": trace.duration_ms,
        "model": trace.model,
        "spans": [
            {"kind": s.kind, "name": s.name, "input": s.input, "output": s.output, "is_error": s.is_error, "duration_ms": s.duration_ms}
            for s in trace.spans
        ],
    }


class RunRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=1000)


class ReplayRequest(BaseModel):
    task_id: str


class ToolRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


def create_app(settings: ServerSettings | None = None, llm_factory: Callable[[], LLM] | None = None) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    registry = build_store_registry()
    tasks = {t["id"]: t for t in load_tasks(TASKS_PATH)}
    limiter = RateLimiter(settings.rate_limit_per_minute)
    budget = DailyBudget(settings.daily_run_limit)
    factory = llm_factory or (lambda: build_llm("claude", model=settings.model, effort=settings.effort))
    app = FastAPI(title="AgentLab", version=__version__)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        if UI_PATH.exists():
            return UI_PATH.read_text(encoding="utf-8")
        return "<h1>AgentLab</h1><p>See <a href='/docs'>/docs</a>.</p>"

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "live": settings.live, "model": settings.model if settings.live else "scripted", "tools": len(registry.names())}

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {
            "live": settings.live,
            "model": settings.model,
            "max_steps": settings.max_steps,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "daily_runs_remaining": budget.remaining if settings.live else 0,
            "tools": [{"name": n, "description": registry.get(n).description, "input_schema": registry.get(n).input_schema} for n in registry.names()],
            "tasks": [{"id": t["id"], "prompt": t["prompt"]} for t in tasks.values()],
        }

    @app.post("/api/replay")
    def replay(req: ReplayRequest) -> dict[str, Any]:
        task = tasks.get(req.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"unknown task {req.task_id!r}")
        trace = Agent(_scripted_llm(task), registry, max_steps=settings.max_steps).run(task["prompt"])
        failures = grade(task, trace)
        return {"mode": "replay", "task_id": task["id"], "prompt": task["prompt"], "passed": not failures, "failures": failures,
                "trace": trace_to_json(trace)}

    @app.post("/api/run")
    def run(req: RunRequest, request: Request) -> dict[str, Any]:
        if not settings.live:
            raise HTTPException(status_code=403, detail="Live mode is off on this deployment (no API key configured). Use Replay instead.")
        if not limiter.allow(client_ip(request)):
            raise HTTPException(status_code=429, detail=f"Rate limit: at most {settings.rate_limit_per_minute} live runs per minute.",
                                headers={"Retry-After": "60"})
        if not budget.take():
            raise HTTPException(status_code=429, detail="The daily budget for live runs is used up. Try Replay, or come back tomorrow.")
        try:
            trace = Agent(factory(), registry, max_steps=settings.max_steps).run(req.task)
        except Exception as exc:  # noqa: BLE001 - surface API failures as a clean 502 instead of a stack trace
            log.exception("live run failed")
            raise HTTPException(status_code=502, detail=f"The model call failed: {type(exc).__name__}") from exc
        return {"mode": "live", "prompt": req.task, "trace": trace_to_json(trace)}

    @app.post("/api/tools/{name}")
    def call_tool(name: str, req: ToolRequest, request: Request) -> dict[str, Any]:
        if name not in registry.names():
            raise HTTPException(status_code=404, detail=f"unknown tool {name!r}")
        result = registry.call(name, req.input)
        return {"tool": name, "input": req.input, "output": result.output, "is_error": result.is_error}

    return app


app = create_app()
