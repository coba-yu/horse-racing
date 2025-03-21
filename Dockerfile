# https://docs.astral.sh/uv/guides/integration/docker/
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Required for selenium
RUN apt-get update && apt-get install -y \
		chromium \
		chromium-driver \
	&& rm -rf /var/lib/apt/lists/*

# Required for lightgbm
RUN apt-get install -y \
		libgomp1

WORKDIR /app
COPY src /app/src
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

RUN uv sync \
	&& mv /app/.venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/app/src
