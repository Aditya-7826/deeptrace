# DeepTrace API: slim CPU image, non-root, healthcheck. Weights are mounted, not baked in.
#   docker compose up --build        (see docker-compose.yml: 2 GB memory limit)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    DEEPTRACE_WEIGHTS_DIR=/app/weights \
    DEEPTRACE_THREADS=2

# OpenCV (headless) still needs glib at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first (cached layer); torch comes from the CPU-only index (pyproject.toml).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 deeptrace \
    && mkdir -p /app/weights \
    && chown deeptrace /app/weights
USER deeptrace

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "deeptrace", "serve", "--host", "0.0.0.0", "--port", "8000"]
