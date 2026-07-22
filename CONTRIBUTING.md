# Contributing

Thanks for looking at the internals. This guide covers how the project is put
together and the workflow it expects.

## Development setup

```bash
uv sync                          # create .venv and install deps + dev tools
docker compose up -d postgres    # tests and the app need pgvector Postgres
uv run cs336-rag ingest          # build the knowledge base (needs OPENAI_KEY)
```

See [docs/setup.md](docs/setup.md) for the full setup, and
[docs/usage.md](docs/usage.md) for the CLI/API surface.

## Quality gate

Everything CI runs is behind `make`:

```bash
make ci            # install + lint + format-check + typecheck + tests
make lint          # ruff check
make format        # ruff format (write)
make typecheck     # mypy (strict)
make test          # pytest (unit + Postgres integration)
```

`make ci` must pass before a change is ready. The integration tests need a
reachable Postgres; they are skipped locally without one but **fail** in CI, so a
missing database can't hide a broken test. `make pre-commit-install` wires the
same checks into a git hook.

## How the code is organised

The package is a thin, layered pipeline; each module has one job and depends only
on the layers below it.

```
config.py        Settings (env-driven) + the SearchMethod / prompt literals
models.py        domain models: TranscriptSegment, VideoTranscript, Chunk
llm.py           OpenAI-compatible client construction + shared retry policy
embeddings.py    batched embedding client (Embedder protocol)
db.py            Postgres/pgvector access; schema init; connection pool
ingest/          transcripts → chunking → pipeline (the offline path)
retrieval.py     text / vector / hybrid / rerank search + query rewriting
rag.py           retrieve → prompt → generate; the prompt variants
service.py       RagService: holds clients, times each phase
conversations.py telemetry persistence (conversations, feedback)
api.py           FastAPI app: /api/ask, /api/feedback, /api/stats, UI
evals/           retrieval + answer evaluations, metrics, ground truth
cli.py           `cs336-rag` entry point wiring the above into commands
```

Two design rules worth knowing:

- **The knowledge base and the telemetry are independent.** `chunks` is rebuilt
  by ingestion; `conversations`/`feedback` are written by the app and read by
  Grafana. Re-ingesting never touches usage history — the schemas are split
  (`schema.sql` vs `schema_app.sql`) precisely so a serving process can create
  its own tables without any chance of dropping `chunks`.
- **The evaluated pipeline is the served pipeline.** The retrieval evaluation
  calls the same `retrieval.search()` the API uses (sharing a precomputed query
  vector), so a "winning" method can't diverge from what actually ships.

## Working style

This repo follows a few conventions consistently; please keep to them.

- **Red/green TDD.** Add a failing test first (a `test:` commit), then make it
  pass (a `feat:`/`fix:` commit). Most modules were built this way and the
  history shows it.
- **Decisions are measured, not asserted.** Retrieval method, prompt variant, and
  generation mode were each chosen by a committed evaluation. If you change one,
  re-run the relevant evaluation and update
  [docs/evaluation.md](docs/evaluation.md) with the numbers.
- **Typed and linted.** `ruff` (a broad rule set) and `mypy --strict` are clean;
  keep them that way rather than adding ignores.
- **Small, reviewed PRs.** One concern per PR, with a short description of what
  and why. External calls are mocked in unit tests; integration tests use a real
  Postgres.

## Adding things

- **A new retrieval method** → add it to `SearchMethod` in `models.py`, implement
  it in `retrieval.py`, handle it in `search()`, and it is automatically included
  in `evaluate-retrieval`.
- **A new prompt variant** → add it to `PROMPT_VARIANTS` and `PromptVariantName`;
  it is picked up by `evaluate-prompts` automatically.
- **A new setting** → add it to `Settings` and forward it in `docker-compose.yml`
  (a test asserts every field is overridable in the container).
- **A new dashboard panel** → edit `grafana/dashboards/cs336-rag.json`; its SQL is
  validated against the real schema by `tests/test_dashboard.py`.
