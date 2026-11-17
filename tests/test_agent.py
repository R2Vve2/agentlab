import json

import pytest

from agentlab.agent import Agent
from agentlab.llm import LLMResponse, ScriptedLLM
from agentlab.store_tools import build_store_registry
from agentlab.tracing import TraceStore


@pytest.fixture
def registry():
    return build_store_registry()


def test_single_tool_round_trip(registry, tmp_path):
    llm = ScriptedLLM([[ScriptedLLM.tool_call("lookup_order", order_id="ORD-1002")], "It has shipped."])
    store = TraceStore(tmp_path / "t.jsonl")
    trace = Agent(llm, registry, trace_store=store).run("status of ORD-1002?")
    assert trace.status == "completed" and trace.final_answer == "It has shipped."
    assert trace.steps == 2 and trace.tool_calls == ["lookup_order"]
    # The second model call must have seen the assistant tool_use turn and the tool_result turn.
    second = llm.calls[1]
    assert second[-2]["role"] == "assistant" and second[-2]["content"][0]["type"] == "tool_use"
    assert second[-1]["role"] == "user" and second[-1]["content"][0]["type"] == "tool_result"
    assert "DHL-77812390" in second[-1]["content"][0]["content"]
    saved = store.load()
    assert len(saved) == 1 and saved[0]["trace_id"] == trace.trace_id and "messages" not in saved[0]


def test_parallel_tool_results_in_one_message(registry):
    llm = ScriptedLLM([[ScriptedLLM.tool_call("today"), ScriptedLLM.tool_call("get_policy", topic="returns")], "done"])
    trace = Agent(llm, registry).run("x")
    results_msg = llm.calls[1][-1]
    assert results_msg["role"] == "user" and len(results_msg["content"]) == 2
    assert [b["tool_use_id"] for b in results_msg["content"]] == [c["id"] for c in llm.calls[1][-2]["content"]]
    assert trace.tool_calls == ["today", "get_policy"]


def test_tool_error_is_reported_not_raised(registry):
    llm = ScriptedLLM([[ScriptedLLM.tool_call("lookup_order", order_id="ORD-9999")], "Not found."])
    trace = Agent(llm, registry).run("x")
    err_block = llm.calls[1][-1]["content"][0]
    assert err_block["is_error"] is True and "not found" in err_block["content"]
    assert trace.spans[1].is_error and trace.status == "completed"


def test_max_steps_guard(registry):
    llm = ScriptedLLM([[ScriptedLLM.tool_call("today")]] * 5)
    trace = Agent(llm, registry, max_steps=3).run("loop forever")
    assert trace.status == "max_steps" and trace.steps == 3


def test_refusal_and_max_tokens_stop_reasons(registry):
    class Stubs:
        model = "scripted"

        def __init__(self, reason):
            self.reason = reason

        def complete(self, system, messages, tools):
            return LLMResponse(self.reason, [{"type": "text", "text": "partial"}], 5, 5)

    assert Agent(Stubs("refusal"), registry).run("x").status == "refused"
    t = Agent(Stubs("max_tokens"), registry).run("x")
    assert t.status == "error" and t.error == "max_tokens"


def test_tool_output_truncated(registry):
    llm = ScriptedLLM([[ScriptedLLM.tool_call("search_products", query="monitor headphones mouse", max_results=8)], "ok"])
    Agent(llm, registry, max_tool_output_chars=50).run("x")
    assert len(llm.calls[1][-1]["content"][0]["content"]) == 50


def test_trace_accounting(registry):
    llm = ScriptedLLM([[ScriptedLLM.tool_call("today")], "ok"])
    trace = Agent(llm, registry).run("x")
    assert trace.input_tokens == 200 and trace.output_tokens == 70 and trace.cost_usd == 0.0
    d = trace.to_dict()
    assert json.dumps(d) and d["steps"] == 2 and d["tool_calls"] == ["today"]
