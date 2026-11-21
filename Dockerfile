FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PORT=8000
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY agentlab ./agentlab
COPY data ./data
COPY ui ./ui
COPY evals/tasks.json ./evals/tasks.json
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health').status==200 else 1)"
# Default: the web app. Set ANTHROPIC_API_KEY to enable live questions.
# CLI instead:  docker run --rm agentlab python -m agentlab eval --min-pass-rate 1.0
CMD ["sh", "-c", "uvicorn agentlab.server:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
