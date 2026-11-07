# Hugging Face Space (Docker SDK, free "CPU basic" hardware: 2 vCPU, 16 GB RAM).
#
# The Space repo holds only this file and README.md. The app itself comes from a tagged
# release of github.com/Aditya-7826/deeptrace and is installed with `uv sync --frozen`, so
# the Space runs exactly the dependency versions in uv.lock. (A plain pip install cannot work:
# facenet-pytorch pins torch<2.3, which DeepTrace overrides in pyproject.toml.)
# Weights are downloaded from that release at startup and checked against SHA-256 sums.
FROM python:3.12-slim

# Set a Space variable DEEPTRACE_REF to deploy another tag (Spaces pass variables as build args).
ARG DEEPTRACE_REF=v1.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /usr/local/bin/uv

# Spaces run the container as uid 1000
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    DEEPTRACE_THREADS=2 \
    DEEPTRACE_WEIGHTS_DIR=/home/user/app/weights
WORKDIR /home/user/app

ADD --chown=user https://github.com/Aditya-7826/deeptrace/archive/refs/tags/${DEEPTRACE_REF}.tar.gz /tmp/src.tar.gz
RUN tar -xzf /tmp/src.tar.gz --strip-components=1 \
    && rm /tmp/src.tar.gz \
    && uv sync --frozen --no-dev --extra space

EXPOSE 7860
CMD ["python", "space/app.py"]
