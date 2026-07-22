# Usage

Three ways to interact with the assistant: the **web UI**, the **HTTP API**, and
the **CLI**. All commands assume the knowledge base has been built
(`cs336-rag ingest`); see [setup.md](setup.md).

Run CLI commands with `uv run cs336-rag <command>` locally, or
`docker compose run --rm app cs336-rag <command>` inside the stack.

## Web UI

Open <http://localhost:8000>. Type a question (or click an example), and you get:

- the answer, with inline `[n]` citations hyperlinked to the exact lecture second;
- a numbered source list;
- a 👍/👎 control that records feedback against that answer;
- the latency and which prompt served it.

## HTTP API

Interactive docs (Swagger UI) are at <http://localhost:8000/docs>.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/api/ask` | Answer a question |
| `POST` | `/api/feedback` | Record 👍/👎 for a conversation |
| `GET` | `/api/stats` | Aggregate counters |
| `GET` | `/` | Web UI |

### Ask a question

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is byte pair encoding and why is it used?"}'
```

```jsonc
{
  "conversation_id": "3b2008df-...",
  "answer": "Byte pair encoding merges the most frequent pair of adjacent tokens [1] ...",
  "variant": "grounded",
  "sources": [
    {"id": "JuoVZkPBiKk:37", "title": "Lecture 1: ...",
     "url": "https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1834s", "start": 1834.0}
  ],
  "retrieval_ms": 1105.2, "generation_ms": 4210.5, "total_ms": 5315.7
}
```

Optional `"variant"` (`baseline` | `grounded` | `tutor`) overrides the default
prompt for that request. A blank question is rejected with `422`.

### Record feedback

Use the `conversation_id` from the answer. `rating` is `1` (👍) or `-1` (👎); a
second vote replaces the first.

```bash
curl -s -X POST http://localhost:8000/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id": "3b2008df-...", "rating": 1}'
```

## CLI

| Command | What it does |
|---|---|
| `fetch-transcripts [--force]` | Download lecture transcripts into `data/raw/` (Whisper fallback) |
| `ingest` | Chunk → embed → load the Postgres knowledge base (idempotent) |
| `ask "<question>" [--variant V]` | Answer one question, printing the answer and sources |
| `serve [--host --port --reload]` | Run the FastAPI app + UI |
| `generate-ground-truth [--sample --per-chunk --seed]` | Build the evaluation question set with the LLM |
| `evaluate-retrieval [--limit]` | Score all retrieval methods (hit rate / MRR) |
| `evaluate-prompts [--sample --seed]` | Judge the prompt variants with the LLM judge |

### One-shot question from the terminal

```bash
uv run cs336-rag ask "How does FlashAttention reduce memory usage?"
```

### Reproduce the evaluations

The datasets and results are committed, so these reproduce the numbers in
[evaluation.md](evaluation.md):

```bash
# retrieval: writes data/eval/retrieval_eval.json, prints the table
uv run cs336-rag evaluate-retrieval

# answer prompts: writes data/eval/answer_eval.json, prints the table + winner
uv run cs336-rag evaluate-prompts --sample 30
```

To regenerate the ground-truth question set from scratch (costs LLM calls):

```bash
uv run cs336-rag generate-ground-truth --sample 150 --per-chunk 2
```

## Monitoring

Feedback and every answered question are logged to Postgres and charted in
Grafana at <http://localhost:3000>. To see a populated dashboard immediately,
seed synthetic telemetry:

```bash
uv run python scripts/seed_demo_data.py --count 400
```

See [monitoring.md](monitoring.md) for the dashboard details.
