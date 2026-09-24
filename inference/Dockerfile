# Image of the batch inference ECS task (last step of the Step Functions pipeline). It needs the
# training package (path dependency), so build from the REPOSITORY ROOT:
#   docker build -f inference/Dockerfile -t inference .
# inference/Dockerfile.dockerignore limits the context to the two projects.
FROM ghcr.io/astral-sh/uv:0.8 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY training/pyproject.toml training/README.md /src/training/
COPY training/src /src/training/src
WORKDIR /src/inference
COPY inference/pyproject.toml inference/uv.lock ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project
COPY inference/README.md ./
COPY inference/src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
COPY --from=build /opt/venv /opt/venv
RUN useradd --create-home --uid 1000 app
USER app
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
CMD ["python", "-m", "steam_inference"]
