import pytest
from fastapi.testclient import TestClient

from agentlab.llm import ScriptedLLM
from agentlab.server import DailyBudget, RateLimiter, ServerSettings, create_app


@pytest.fixture
def replay_client():
    with TestClient(create_app(ServerSettings(live=False))) as c:
        yield c


def live_client(**overrides):
    def factory():
        return ScriptedLLM([[ScriptedLLM.tool_call("lookup_order", order_id="ORD-1002")], "It has shipped with DHL."])

    settings = ServerSettings(live=True, rate_limit_per_minute=overrides.get("rate", 5), daily_run_limit=overrides.get("daily", 100))
    return TestClient(create_app(settings, llm_factory=factory))


def test_ui_and_info(replay_client):
    assert "AgentLab" in replay_client.get("/").text
    info = replay_client.get("/api/info").json()
    assert info["live"] is False and len(info["tasks"]) == 12
    assert {t["name"] for t in info["tools"]} >= {"lookup_order", "calculate"}
    assert replay_client.get("/health").json()["model"] == "scripted"


def test_replay_runs_real_tools_and_grades(replay_client):
    r = replay_client.post("/api/replay", json={"task_id": "order_total_check"}).json()
    assert r["passed"] is True and r["failures"] == []
    tools = [s for s in r["trace"]["spans"] if s["kind"] == "tool"]
    assert [s["name"] for s in tools] == ["lookup_order", "calculate"]
    assert "226.99" in tools[1]["output"]
    assert replay_client.post("/api/replay", json={"task_id": "nope"}).status_code == 404


def test_live_disabled_without_key(replay_client):
    r = replay_client.post("/api/run", json={"task": "status of ORD-1002?"})
    assert r.status_code == 403 and "Live mode is off" in r.json()["detail"]


def test_live_run_rate_limit_and_budget():
    with live_client(rate=2, daily=3) as c:
        h = {"X-Forwarded-For": "203.0.113.9"}
        ok = c.post("/api/run", json={"task": "status of ORD-1002?"}, headers=h)
        assert ok.status_code == 200
        body = ok.json()
        assert body["mode"] == "live" and body["trace"]["final_answer"] == "It has shipped with DHL."
        assert body["trace"]["tool_calls"] == ["lookup_order"]
        assert c.post("/api/run", json={"task": "again please"}, headers=h).status_code == 200
        assert c.post("/api/run", json={"task": "third time"}, headers=h).status_code == 429  # per-IP limit
        other = {"X-Forwarded-For": "198.51.100.4"}
        assert c.post("/api/run", json={"task": "from elsewhere"}, headers=other).status_code == 200
        exhausted = c.post("/api/run", json={"task": "one more"}, headers={"X-Forwarded-For": "192.0.2.1"})
        assert exhausted.status_code == 429 and "daily budget" in exhausted.json()["detail"]


def test_tool_playground(replay_client):
    ok = replay_client.post("/api/tools/calculate", json={"input": {"expression": "6*7"}}).json()
    assert ok["output"] == "42" and ok["is_error"] is False
    bad = replay_client.post("/api/tools/lookup_order", json={"input": {"wrong": 1}}).json()
    assert bad["is_error"] is True
    assert replay_client.post("/api/tools/rm_rf", json={"input": {}}).status_code == 404


def test_limiter_and_budget_units():
    rl = RateLimiter(1)
    assert rl.allow("a", now=0) and not rl.allow("a", now=1) and rl.allow("a", now=61)
    day = {"d": "2026-11-21"}
    b = DailyBudget(1, today=lambda: day["d"])
    assert b.take() and not b.take() and b.remaining == 0
    day["d"] = "2026-11-22"
    assert b.take()
