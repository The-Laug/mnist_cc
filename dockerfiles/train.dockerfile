# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only lock/project files first so the dependency layer is cacheable
COPY uv.lock pyproject.toml ./

# Use a persistent cache for uv; keep --locked to respect the lockfile
ENV UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache \
    uv sync --locked

# App sources (changes here won't invalidate the deps layer)
COPY src/ src/
COPY data/ data/

# Create output directories
RUN mkdir -p models reports/figures

ENTRYPOINT ["uv", "run", "src/mnist_cc/train.py"]