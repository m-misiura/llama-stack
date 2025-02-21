FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files first for better layer caching
COPY pyproject.toml README.md ./
COPY llama_stack ./llama_stack

# Install dependencies from pyproject.toml and additional server requirements
RUN pip install --no-cache-dir -e .[distribution,server,telemetry] && \
    pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        python-multipart \
        opentelemetry-api \
        opentelemetry-sdk \
        opentelemetry-instrumentation-fastapi \
        opentelemetry-exporter-otlp \
        opentelemetry-exporter-otlp-proto-http \
        aiosqlite \
        openai

# Copy distribution files
COPY distributions/remote-vllm-fms/two-detectors/run.yaml /root/llamastack-run-remote-vllm.yaml

EXPOSE 5001

ENTRYPOINT ["python", "-m", "llama_stack.distribution.server.server", "--yaml-config", "/root/llamastack-run-remote-vllm.yaml"]