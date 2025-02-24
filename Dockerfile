# https://docs.astral.sh/uv/guides/integration/docker/
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

RUN uv sync \
    && mv /app/.venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/app/src
