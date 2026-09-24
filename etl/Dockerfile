# Image for the ECS `dbt` task: `python -m steam_etl` runs `dbt build` on Athena.
# Build from etl/: docker build -t etl .
FROM ghcr.io/astral-sh/uv:0.8 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 etl
COPY --from=build /opt/venv /opt/venv
COPY --chown=etl dbt /app/dbt
# dbt writes its manifest / logs to a writable path, not into the project.
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 \
    DBT_PROJECT_DIR=/app/dbt DBT_TARGET_PATH=/tmp/dbt/target DBT_LOG_PATH=/tmp/dbt/logs \
    DBT_SEND_ANONYMOUS_USAGE_STATS=false
USER etl
CMD ["python", "-m", "steam_etl"]
