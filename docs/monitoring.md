# Monitoring

The application logs every answered question to Postgres (`conversations`) and
records user votes (`feedback`). Grafana reads those tables directly through a
provisioned datasource and renders a dashboard — no extra pipeline.

## Running it

```bash
docker compose up -d postgres grafana
```

Open <http://localhost:3000>. The dashboard **CS336 Lecture Assistant —
Monitoring** is provisioned automatically. Log in with `admin` / `admin` by
default, or override those credentials with `GRAFANA_USER` /
`GRAFANA_PASSWORD`.

Grafana is bound to `127.0.0.1` by default. For a local demo where anonymous
viewing is acceptable, set `GRAFANA_ANONYMOUS_ENABLED=true`; do not expose that
mode on a shared network because viewers can query the provisioned datasource.

Everything is provisioned from version-controlled files, so the dashboard is
reproducible from a clean checkout:

- `grafana/provisioning/datasources/postgres.yml` — the Postgres datasource
- `grafana/provisioning/dashboards/dashboards.yml` — the dashboard loader
- `grafana/dashboards/cs336-rag.json` — the dashboard itself

## Populating it

Real usage fills the tables as you ask questions in the UI. To see the
dashboard populated immediately (e.g. for a demo or screenshots), seed
synthetic telemetry — this only writes to the telemetry tables, never the
knowledge base:

```bash
uv run python scripts/seed_demo_data.py --count 400
```

## Panels

Eleven panels (the rubric asks for at least five charts):

| Panel | Type | What it shows |
|---|---|---|
| Total conversations | stat | Lifetime question count |
| Feedback rate | stat | Share of answers that got a vote |
| Satisfaction | stat | Share of votes that were 👍 |
| Median answer latency | stat | p50 of `total_ms` |
| Conversations over time | time series | Volume trend |
| Answer latency (p50 / p95) | time series | Latency trend and tail |
| Token usage over time | time series | Avg prompt / completion tokens per answer |
| Latency breakdown | stacked time series | Retrieval vs generation split |
| Feedback breakdown | donut | 👍 vs 👎 |
| Prompt variant usage | bar | Which prompt served each answer |
| Retrieval method usage | bar | Which retrieval method served each answer |

The panel SQL is validated in CI: `tests/test_dashboard.py` runs every panel's
query against a real Postgres, so a renamed column breaks the build instead of
silently emptying a chart.
