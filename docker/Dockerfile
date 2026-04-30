# Multi-stage Dockerfile for the SDET Brain server.
#
# Stage 1 (builder): install dependencies into a cached venv via uv.
# Stage 2 (runtime): copy the venv plus source, drop privileges, run uvicorn.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/

RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python -e .

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY pyproject.toml /app/

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8080/health').raise_for_status()" || exit 1

CMD ["sdet-brain-server"]
