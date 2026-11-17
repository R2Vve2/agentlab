from pathlib import Path

from agentlab.evals import format_report, grade, load_tasks, run_eval
from agentlab.store_tools import build_store_registry
from agentlab.tracing import Span, Trace

TASKS = Path(__file__).resolve().parents[1] / "evals" / "tasks.json"


def _trace(answer: str, tools: list[str], status: str = "completed", steps: int = 2) -> Trace:
    t = Trace(task="t", model="scripted", status=status, final_answer=answer)
    t.spans = [Span("llm", "scripted", {}, {}, 1.0)] * steps + [Span("tool", n, {}, "", 1.0) for n in tools]
    return t


def test_graders():
    task = {"expected_contains": ["yes"], "expected_number": 226.99, "must_call": ["lookup_order"], "must_not_call": ["today"], "max_steps": 3}
    assert grade(task, _trace("Yes, total is 226.99", ["lookup_order"])) == []
    fails = grade(task, _trace("total is 1,226.99", ["today"], steps=4))
    assert any("missing text" in f for f in fails)
    assert any("number 226.99 not found" in f for f in fails)  # 1,226.99 != 226.99
    assert any("lookup_order not called" in f for f in fails) and any("today must not" in f for f in fails)
    assert any("steps > max" in f for f in fails)
    assert grade({}, _trace("", [], status="max_steps")) == ["status=max_steps"]


def test_number_grader_handles_commas_and_tolerance():
    assert grade({"expected_number": 1226.99}, _trace("It is 1,226.99.", [])) == []
    assert grade({"expected_number": 34, "tolerance": 1}, _trace("about 33 months", [])) == []


def test_scripted_suite_passes_end_to_end(tmp_path):
    tasks = load_tasks(TASKS)
    report = run_eval(tasks, build_store_registry())
    text = format_report(report)
    failed = [r for r in report["results"] if not r["passed"]]
    assert not failed, text
    assert report["summary"]["pass_rate"] == 1.0 and report["summary"]["tasks"] == len(tasks)
    assert "PASS  order_status_simple" in text


def test_only_filter_and_failed_task_does_not_abort():
    tasks = load_tasks(TASKS)
    tasks[0] = {**tasks[0], "script": ["wrong answer"]}
    report = run_eval(tasks, build_store_registry(), only={tasks[0]["id"], tasks[1]["id"]})
    assert report["summary"]["tasks"] == 2 and report["summary"]["passed"] == 1
    assert report["results"][0]["failures"]
