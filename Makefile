.PHONY: install test lint eval eval-claude run mcp docker-build

install:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

eval:
	python -m agentlab eval --min-pass-rate 1.0

eval-claude:
	python -m agentlab eval --llm claude --model claude-opus-5-5 --out evals/results_claude.json

run:
	python -m agentlab run "$(TASK)"

mcp:
	python -m agentlab.mcp_server

docker-build:
	docker build -t agentlab:latest .
