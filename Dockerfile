FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY agentlab ./agentlab
COPY data ./data
COPY evals/tasks.json ./evals/tasks.json
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
# Default: run the offline, deterministic evaluation suite. Override to run the real agent:
#   docker run -e ANTHROPIC_API_KEY=... agentlab run "What is the status of order ORD-1002?"
ENTRYPOINT ["python", "-m", "agentlab"]
CMD ["eval", "--min-pass-rate", "1.0"]
