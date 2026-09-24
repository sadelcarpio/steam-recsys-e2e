# Image for both ECS scraping tasks. The task definition picks the module:
#   games:   ["python", "-m", "steam_ingestion.games_scraping"]
#   reviews: ["python", "-m", "steam_ingestion.reviews_scraping"]
# Build from data_ingestion/: docker build -f scraping.Dockerfile -t data-ingestion .
FROM ghcr.io/astral-sh/uv:0.8 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra scraping --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra scraping --no-editable

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 scraper
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
USER scraper
CMD ["python", "-m", "steam_ingestion.games_scraping"]
