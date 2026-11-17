"""Command-line interface.

    python -m agentlab run "What is the status of order ORD-1002?"      # needs ANTHROPIC_API_KEY
    python -m agentlab eval                                             # scripted mode (offline, deterministic)
    python -m agentlab eval --llm claude --model claude-opus-5-5        # real model
    python -m agentlab traces                                           # summarise recorded traces
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agentlab.agent import Agent
from agentlab.evals import format_report, load_tasks, run_eval
from agentlab.llm import build_llm
from agentlab.store_tools import build_store_registry
from agentlab.tracing import TraceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default=None, choices=[None, "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--traces", default="traces/traces.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one task with the real model")
    p_run.add_argument("task")
    p_run.add_argument("--max-steps", type=int, default=10)

    p_eval = sub.add_parser("eval", help="run the evaluation suite")
    p_eval.add_argument("--tasks", default="evals/tasks.json")
    p_eval.add_argument("--llm", default="scripted", choices=["scripted", "claude"])
    p_eval.add_argument("--only", nargs="*", default=None, help="task ids to run")
    p_eval.add_argument("--out", default="evals/results.json")
    p_eval.add_argument("--min-pass-rate", type=float, default=0.0)

    sub.add_parser("traces", help="summarise recorded traces")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    registry = build_store_registry()
    store = TraceStore(args.traces)

    if args.command == "run":
        llm = build_llm("claude", model=args.model, effort=args.effort)
        trace = Agent(llm, registry, max_steps=args.max_steps, trace_store=store).run(args.task)
        for span in trace.spans:
            if span.kind == "tool":
                print(f"  -> {span.name}({json.dumps(span.input)}) = {str(span.output)[:120]}{' [error]' if span.is_error else ''}")
        print(f"\n{trace.final_answer}\n")
        print(f"[{trace.status}] steps={trace.steps} tokens={trace.input_tokens}+{trace.output_tokens} cost=${trace.cost_usd} trace={trace.trace_id}")
        return 0

    if args.command == "eval":
        tasks = load_tasks(args.tasks)
        llm = build_llm("claude", model=args.model, effort=args.effort) if args.llm == "claude" else None
        report = run_eval(tasks, registry, llm=llm, trace_store=store, only=set(args.only) if args.only else None)
        print(format_report(report))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
        if report["summary"]["pass_rate"] < args.min_pass_rate:
            print(f"FAIL: pass rate {report['summary']['pass_rate']} < {args.min_pass_rate}", file=sys.stderr)
            return 1
        return 0

    traces = store.load()
    if not traces:
        print("no traces recorded yet")
        return 0
    print(f"{len(traces)} traces in {store.path}")
    for t in traces[-20:]:
        print(f"{t['trace_id']}  {t['status']:<10} steps={t['steps']}  tools={','.join(t['tool_calls']) or '-'}  cost=${t['cost_usd']}  {t['task'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
