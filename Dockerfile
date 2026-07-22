# syntax=docker/dockerfile:1

# ---- builder: resolve and install dependencies with uv ----
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install third-party dependencies first, in their own cached layer, so a
# source change does not re-resolve the whole environment.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then install the project itself.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime: slim image with just the venv and the app ----
FROM python:3.12-slim AS runtime

# psycopg[binary] ships its own libpq, so no system libs are needed.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# No curl in the slim image; probe with the interpreter we already have.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else sys.exit(1)"]

CMD ["cs336-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
