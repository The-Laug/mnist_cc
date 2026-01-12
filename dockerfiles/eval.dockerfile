# Evaluation image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer
COPY uv.lock pyproject.toml ./
RUN uv sync --locked --frozen

# Application code and data
COPY src/ src/
COPY data/ data/

# Default model path can be overridden at runtime
ENV MODEL_PATH=models/model.pth

# Use a shell so MODEL_PATH is expanded at runtime
ENTRYPOINT ["sh", "-c", "uv run invoke evaluate --model-checkpoint $MODEL_PATH"]
