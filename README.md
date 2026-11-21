# AgentLab — tool-using LLM agents with tracing and evals

[![CI](https://github.com/R2Vve2/agentlab/actions/workflows/ci.yml/badge.svg)](https://github.com/R2Vve2/agentlab/actions/workflows/ci.yml)

A small, readable agent framework built directly on the Claude Messages API: **typed tools**, an explicit
**agent loop**, **structured traces** with token and cost accounting, a **deterministic evaluation harness**,
and an **MCP server** that exposes the same tools to any MCP client. The demo scenario is a customer-support
agent for an online electronics store (orders, catalogue, policies, date and money arithmetic).

```
 user task ─► Agent loop ─► Claude (tools declared from the registry) ─► tool_use blocks
                 ▲                                                            │
                 └── tool_result blocks (all results in ONE user turn) ◄── ToolRegistry.call()
                                   │                                        (pydantic-validated, errors returned, never raised)
                                   ▼
                      Trace: spans for every model + tool call, tokens, cost, status ─► traces/traces.jsonl
                                   ▼
                      Evals: graders (contains / number / must_call / must_not_call / max_steps) ─► pass rate, cost
```

## Why this project

| Hiring-market skill | Where it shows up here |
|---|---|
| Agentic LLM engineering | Hand-written ReAct loop over the Messages API: parallel tool calls, `is_error` tool results, `pause_turn`, `max_tokens` and `refusal` handling, step budget |
| Tool design | JSON Schema generated from type hints, `strict` schemas, pydantic validation of model inputs, safe arithmetic (AST, no `eval`) |
| Observability | Every run becomes a trace of spans with latency, tokens and USD estimate; JSONL export; CLI summary |
| Evaluation-driven development | 12-task suite with deterministic graders; scripted mode runs offline in CI, `--llm claude` measures the real model with the same graders |
| MCP / interoperability | `python -m agentlab.mcp_server` serves the registry over the Model Context Protocol for Claude Desktop, Claude Code or IDEs |
| Testing & CI | 17 tests (loop, tools, graders, MCP), ruff, Docker image whose default command is the eval suite |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m agentlab eval                   # offline, deterministic: replays ideal trajectories through the real loop + tools
export ANTHROPIC_API_KEY=...              # Windows: set ANTHROPIC_API_KEY=...
python -m agentlab run "Sara (sara@example.com) wants to return the headphones from ORD-1001 today. Is she within the return window?"
python -m agentlab eval --llm claude      # measure Claude on the same 12 tasks
python -m agentlab traces                 # list recorded traces
```

Example run with the real model (illustrative transcript showing the output format):

```
  -> lookup_order({"order_id": "ORD-1001"}) = {"customer_email": "sara@example.com", "status": "delivered", "delivered_on": "2026-11-05", ...
  -> get_policy({"topic": "returns"}) = Items can be returned within 30 days of delivery ... Opened audio products ... within 14 days ...
  -> today({}) = 2026-11-17
  -> days_between({"start": "2026-11-05", "end": "2026-11-17"}) = 12

Yes. The headphones were delivered on 2026-11-05 (12 days ago); opened audio products can be returned within 14 days, so she is inside the window.

[completed] steps=3 tokens=4210+312 cost=$0.02308 trace=4f1c9e2a7b10
```

### Web app

```bash
uvicorn agentlab.server:app --reload       # http://localhost:8000
```

The page has three parts:

- **Replay:** runs any of the 12 graded tasks through the real agent loop and real tools, with `ScriptedLLM` playing the
  model, then shows the answer, the grade and every step of the trace. Free, no API key.
- **Live questions:** when `ANTHROPIC_API_KEY` is set, visitors ask their own questions and Claude drives the agent.
  A per-visitor rate limit (`AGENTLAB_RATE_LIMIT_PER_MINUTE`, default 5) and a daily cap (`AGENTLAB_DAILY_RUN_LIMIT`,
  default 100) bound the cost of a public deployment. `AGENTLAB_MODEL` and `AGENTLAB_EFFORT` choose the model and effort.
- **Tool playground:** calls one tool directly with JSON input, including invalid input to show error handling.

To host it for free, create a Render web service from this public repository (Docker runtime, Free instance) or apply
`render.yaml` as a Blueprint. Add `ANTHROPIC_API_KEY` as a secret only if you want live questions and accept the cost.

### Docker

```bash
docker build -t agentlab .
docker run --rm -p 8000:8000 agentlab                            # web app on http://localhost:8000
docker run --rm agentlab python -m agentlab eval --min-pass-rate 1.0   # offline eval suite
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY agentlab python -m agentlab run "What is the status of ORD-1002?"
```

## How it works

### Tools (`agentlab/tools.py`)

```python
registry = ToolRegistry()

@registry.tool("Number of days between two ISO dates.", params={"start": "Start date", "end": "End date"})
def days_between(start: str, end: str) -> int:
    ...
```

The decorator builds the `input_schema` from the signature (`str/int/float/bool`, defaults become optional),
and `registry.call(name, input)` validates the model's arguments with pydantic. Unknown tools, invalid
arguments and exceptions all come back as `ToolResult(is_error=True)` so the model can read the message and
recover (see the `unknown_order_recovery` and `bad_tool_input_recovery` eval tasks).

### Agent loop (`agentlab/agent.py`)

1. Call the model with the system prompt, the conversation and `registry.to_api()`.
2. Append the assistant content unchanged (thinking blocks included).
3. If there are `tool_use` blocks, run **all** of them and return **all** results in a single user message,
   which is what keeps parallel tool calling working.
4. Stop on `end_turn` (no tool calls), `refusal`, `max_tokens`, or when `max_steps` is reached.

The Claude backend uses `claude-opus-5-5` by default with the server-side refusal fallback enabled, so a
safety-classifier decline is retried on a fallback model inside the same request. `--effort low|medium|high`
tunes reasoning depth and cost.

### Tracing (`agentlab/tracing.py`)

Each run produces a `Trace` with one `Span` per model call and per tool call (name, input, output, latency,
tokens, error flag), plus totals and a cost estimate from a small price table. Traces are appended to
`traces/traces.jsonl`; `agentlab traces` summarises them. Diffing traces between prompt versions is the
fastest way to see *why* a pass rate moved.

### Evals (`agentlab/evals.py`, `evals/tasks.json`)

Each task carries a prompt, graders and a `script` describing the ideal trajectory:

```json
{"id": "return_window_open",
 "prompt": "Sara wants to return the headphones from order ORD-1001 today. Is she still within the return window?",
 "expected_contains": ["yes"], "must_call": ["lookup_order", "get_policy", "days_between"],
 "script": [[{"tool": "lookup_order", "input": {"order_id": "ORD-1001"}}, {"tool": "get_policy", "input": {"topic": "returns"}}],
            [{"tool": "today"}], [{"tool": "days_between", "input": {"start": "2026-11-05", "end": "2026-11-17"}}],
            "Yes. Delivered 2026-11-05, 12 days ago; opened audio products can be returned within 14 days."]}
```

Scripted mode (`ScriptedLLM`) replays the script through the real loop, tools and graders, giving a
zero-cost regression suite for everything except the model. Claude mode swaps in the real model and reports
pass rate, average steps, tokens and estimated cost. The CI gate requires 12/12 in scripted mode.

### MCP server (`agentlab/mcp_server.py`)

```bash
pip install "mcp<2" && python -m agentlab.mcp_server
```

Registers every tool from the registry on a `FastMCP` stdio server, so the identical tool implementations can be
used from Claude Desktop or Claude Code.

## Project layout

| Path | Purpose |
|---|---|
| `agentlab/tools.py` | `ToolRegistry`, schema generation, validation, error capture |
| `agentlab/agent.py` | The agent loop and default system prompt |
| `agentlab/llm.py` | `ClaudeLLM` (Anthropic SDK) and `ScriptedLLM` (offline replay) |
| `agentlab/tracing.py` | `Trace`/`Span`, JSONL store, cost estimates |
| `agentlab/evals.py` | Graders, eval runner, report formatting |
| `agentlab/store_tools.py` | The seven demo tools over `data/*.json` |
| `agentlab/mcp_server.py` | MCP exposure of the registry |
| `agentlab/cli.py` | `run`, `eval`, `traces` commands |
| `evals/tasks.json` | 12 graded tasks with ideal trajectories |
| `tests/` | 17 tests, all offline |

## Roadmap

- LLM-as-judge grader for free-form answers
- Conversation memory with summarisation for long sessions
- Prompt-caching breakpoints on the tool list and system prompt
- Streaming output in the CLI
