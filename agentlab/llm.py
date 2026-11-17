"""Model backends behind one tiny interface.

``ClaudeLLM`` talks to the Claude API. ``ScriptedLLM`` replays a predefined sequence of
responses, which makes the agent loop, the tools and the evaluation harness fully testable
offline and deterministic in CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    stop_reason: str  # end_turn | tool_use | max_tokens | refusal | pause_turn
    content: list[dict[str, Any]]  # API-shaped content blocks (text / tool_use / thinking ...)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class LLM(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class ScriptedLLM:
    """Replays scripted turns. Each turn is either a final text answer or a list of tool calls."""

    model = "scripted"

    def __init__(self, turns: list[str | list[dict[str, Any]]]):
        self._turns = list(turns)
        self.calls: list[list[dict[str, Any]]] = []  # messages seen on every call (for assertions)

    @staticmethod
    def tool_call(name: str, **kwargs: Any) -> dict[str, Any]:
        return {"name": name, "input": kwargs}

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self._turns:
            return LLMResponse("end_turn", [{"type": "text", "text": "(script exhausted)"}], 10, 5)
        turn = self._turns.pop(0)
        if isinstance(turn, str):
            return LLMResponse("end_turn", [{"type": "text", "text": turn}], 100, 30)
        blocks = [
            {"type": "tool_use", "id": f"toolu_{len(self.calls)}_{i}", "name": c["name"], "input": c["input"]}
            for i, c in enumerate(turn)
        ]
        return LLMResponse("tool_use", blocks, 100, 40)


class ClaudeLLM:
    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 8192, effort: str | None = None):
        import anthropic  # lazy import: only needed for real runs

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        # Server-side refusal fallback: a safety-classifier decline is re-run on a fallback
        # model inside the same call instead of ending the agent run.
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
        content = [block.model_dump(exclude_none=True) for block in response.content]
        return LLMResponse(
            stop_reason=response.stop_reason or "end_turn",
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )


def build_llm(kind: str, model: str = "claude-opus-5-5", effort: str | None = None) -> LLM:
    if kind == "claude":
        return ClaudeLLM(model=model, effort=effort)
    raise ValueError(f"unknown LLM backend {kind!r}; use 'claude' (ScriptedLLM is constructed directly with its turns)")
