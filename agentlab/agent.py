"""The agent loop.

A plain, explicit ReAct-style loop over the Messages API: call the model with the tool
definitions, execute every tool it asked for (results returned together in one user turn,
which is what keeps parallel tool use working), append, repeat until the model stops
calling tools or a budget is hit. Every call is recorded as a span in a ``Trace``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentlab.llm import LLM
from agentlab.tools import ToolRegistry
from agentlab.tracing import Span, Trace, TraceStore

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = """You are a careful assistant for an online electronics store's support team.
Use the available tools to look up facts before answering; never guess order details, prices or policies.
When a tool returns an error, read it and decide whether to retry with corrected input or explain the problem.
Keep final answers short and specific, and state numbers exactly as the tools returned them."""


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = 10,
        max_tool_output_chars: int = 4000,
        trace_store: TraceStore | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.system = system
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.trace_store = trace_store

    def run(self, task: str, history: list[dict[str, Any]] | None = None) -> Trace:
        messages: list[dict[str, Any]] = list(history or []) + [{"role": "user", "content": task}]
        trace = Trace(task=task, model=self.llm.model)
        tool_defs = self.tools.to_api()
        try:
            for _step in range(self.max_steps):
                t0 = time.perf_counter()
                response = self.llm.complete(self.system, messages, tool_defs)
                trace.spans.append(
                    Span("llm", self.llm.model, {"messages": len(messages)}, {"stop_reason": response.stop_reason, "text": response.text},
                         round((time.perf_counter() - t0) * 1000, 1), input_tokens=response.input_tokens, output_tokens=response.output_tokens)
                )
                # Pass the assistant turn back unchanged (including any thinking blocks).
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "refusal":
                    trace.status, trace.final_answer = "refused", response.text or "The request was declined."
                    break
                if response.stop_reason == "pause_turn":
                    continue  # server-side tool turn paused; re-send to let it continue
                if response.stop_reason == "max_tokens":
                    trace.status, trace.final_answer = "error", "Response was cut off (max_tokens)."
                    trace.error = "max_tokens"
                    break
                if not response.tool_uses:
                    trace.status, trace.final_answer = "completed", response.text
                    break

                results = []
                for call in response.tool_uses:
                    t0 = time.perf_counter()
                    result = self.tools.call(call["name"], call.get("input") or {})
                    output = result.output[: self.max_tool_output_chars]
                    trace.spans.append(Span("tool", call["name"], call.get("input"), output, round((time.perf_counter() - t0) * 1000, 1), result.is_error))
                    results.append({"type": "tool_result", "tool_use_id": call["id"], "content": output, "is_error": result.is_error})
                messages.append({"role": "user", "content": results})
            else:
                trace.status = "max_steps"
                trace.final_answer = f"Stopped after {self.max_steps} steps without a final answer."
        except Exception as exc:  # noqa: BLE001 - record the failure in the trace, then re-raise
            trace.status, trace.error = "error", f"{type(exc).__name__}: {exc}"
            if self.trace_store:
                self.trace_store.append(trace)
            raise
        trace.messages = messages  # handy for multi-turn callers
        if self.trace_store:
            self.trace_store.append(trace)
        return trace
