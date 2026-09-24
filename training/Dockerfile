# Image for the SageMaker training jobs: the launcher sets the entry point to
# `python -m steam_training {train,promote}`. Build from training/: docker build -t training .
# TORCH_VARIANT picks the torch extra: cpu (default, ~1.8 GB) or cu128 (GPU instances, DEVICE=auto
# finds CUDA): docker build --build-arg TORCH_VARIANT=cu128 -t training:<sha>-cu128 .
FROM ghcr.io/astral-sh/uv:0.8 AS uv

FROM python:3.12-slim AS build
ARG TORCH_VARIANT=cpu
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra ${TORCH_VARIANT} --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra ${TORCH_VARIANT} --no-editable

FROM python:3.12-slim
COPY --from=build /opt/venv /opt/venv
# Runs as root: SageMaker writes /opt/ml (failure reason, output) with the container user.
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /opt/ml/code
CMD ["python", "-m", "steam_training", "train"]
