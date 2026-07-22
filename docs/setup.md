# Setup

Two ways to run it: the **Docker stack** (recommended — one command brings up
everything) or a **local dev environment** with `uv` for working on the code.

## Prerequisites

- An API key for the OpenAI-compatible endpoint (the project defaults to
  `https://api.nan.builders/v1`, which serves the chat, embedding, rerank, judge
  and Whisper models used here). Put it in `.env` as `OPENAI_KEY`.
- **Docker** with Compose v2, *or* for local dev: [uv](https://docs.astral.sh/uv/)
  and Docker (Postgres still runs in a container).

## Configuration

All configuration is environment variables, read from `.env` (see
[`.env.example`](../.env.example) for the full list with defaults). The only
required value is `OPENAI_KEY`:

```bash
cp .env.example .env
# edit .env and set OPENAI_KEY=sk-...
```

Every setting has a sensible default, and every one is overridable both locally
and inside the Docker stack (compose forwards them all to the app container).
Notable ones:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_KEY` | — (required) | API key for the model endpoint |
| `LLM_BASE_URL` | `https://api.nan.builders/v1` | OpenAI-compatible endpoint |
| `RETRIEVAL_METHOD` | `vector` | Retrieval method served (evaluation winner) |
| `RAG_PROMPT_VARIANT` | `grounded` | Answer prompt served (evaluation winner) |
| `CHAT_DISABLE_THINKING` | `true` | Disable the chat model's reasoning mode (latency) |
| `EMBEDDING_DIM` | `1024` | Embedding width; **must match the ingested data** |
| `DB_*` | `cs336` / `cs336_rag` | Postgres connection |

## Option A — Docker stack (recommended)

```bash
docker compose up -d          # postgres + app + grafana
```

On first run the knowledge base is empty. Populate it from the committed
transcripts (needs `OPENAI_KEY`, embeds ~800 chunks, ~1 minute):

```bash
docker compose run --rm app cs336-rag ingest
```

That's it:

- Web UI → <http://localhost:8000>
- API docs → <http://localhost:8000/docs>
- Grafana → <http://localhost:3000> (see [monitoring.md](monitoring.md))

The app creates its own telemetry tables on startup; ingestion creates and fills
the knowledge base. Both are idempotent, so the commands are safe to re-run.

To rebuild the app image after code changes: `docker compose up -d --build`.

## Option B — Local development

Postgres still runs in Docker; the app and tooling run on the host via `uv`.

```bash
uv sync                          # create .venv and install everything
docker compose up -d postgres    # pgvector Postgres on localhost:5432

uv run cs336-rag ingest          # build the knowledge base
uv run cs336-rag serve           # http://localhost:8000
```

### Quality checks (same as CI)

```bash
make ci            # install + lint + format-check + typecheck + tests
# or individually:
make lint          # ruff check
make format-check  # ruff format --check
make typecheck     # mypy (strict)
make test          # pytest (unit + integration)
```

The integration tests need a reachable Postgres; without one they are skipped
locally but **fail** in CI (so a missing database can't hide a broken test).
Install pre-commit hooks with `make pre-commit-install`.

## Rebuilding the dataset from scratch (optional)

The transcripts are committed under `data/raw/`, so you never need this to run
the project. To re-fetch them from YouTube (e.g. if the playlist changes):

```bash
uv run cs336-rag fetch-transcripts   # writes data/raw/NNN-<video_id>.json
```

Manual English captions are preferred; videos without them fall back to
Whisper transcription of the audio.

## Troubleshooting

- **`OPENAI_KEY` error on `docker compose up`** — the app service requires it;
  set it in `.env`. (Bringing up only `postgres`/`grafana` does not need it.)
- **Answers are empty / "no context"** — the knowledge base is empty; run the
  `ingest` command above.
- **Dimension mismatch on ingest** — you changed `EMBEDDING_DIM` after a previous
  ingest. `init_schema` recreates the `chunks` table automatically on the next
  `ingest`; just re-run it.
- **Grafana bind-mount error on WSL / Docker Desktop** — a stale mount cache;
  `docker compose rm -sf grafana && docker compose up -d grafana`.
