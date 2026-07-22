"""Validate the provisioned Grafana dashboard.

The dashboard is JSON, but its panels contain SQL that must stay in sync
with the telemetry schema. These tests parse the dashboard and run every
panel query against a real Postgres so a renamed column or a typo breaks
the build instead of silently emptying a chart.
"""

import json
import re
from pathlib import Path

import psycopg
import pytest

DASHBOARD = Path(__file__).resolve().parent.parent / "grafana" / "dashboards" / "cs336-rag.json"

# Grafana macros, replaced with valid SQL so the query can be EXPLAINed.
_MACROS = {
    r"\$__timeFilter\(([^)]+)\)": r"\1 BETWEEN now() - interval '7 days' AND now()",
}


def _load() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _panel_queries() -> list[tuple[str, str]]:
    panels = _load()["panels"]
    queries = []
    for panel in panels:
        for target in panel.get("targets", []):
            sql = target.get("rawSql")
            if sql:
                queries.append((panel["title"], sql))
    return queries


def _strip_macros(sql: str) -> str:
    for pattern, replacement in _MACROS.items():
        sql = re.sub(pattern, replacement, sql)
    return sql


def test_dashboard_is_valid_json() -> None:
    dashboard = _load()
    assert dashboard["uid"] == "cs336-rag-monitoring"
    assert dashboard["title"]


def test_has_at_least_five_charts() -> None:
    chart_types = {"timeseries", "piechart", "barchart", "stat", "gauge", "table"}
    charts = [p for p in _load()["panels"] if p["type"] in chart_types]
    assert len(charts) >= 5


def test_every_panel_targets_the_provisioned_datasource() -> None:
    for panel in _load()["panels"]:
        for target in panel.get("targets", []):
            assert target["datasource"]["uid"] == "cs336-postgres"


def test_panel_ids_are_unique() -> None:
    ids = [panel["id"] for panel in _load()["panels"]]
    assert len(ids) == len(set(ids))


def test_there_are_panel_queries_to_validate() -> None:
    # guards the parametrized SQL check below: if the dashboard file moves or
    # its shape changes, _panel_queries() returns [] and those tests silently
    # vanish (0 params). This keeps that from passing unnoticed.
    assert len(_panel_queries()) >= 8


@pytest.mark.integration
@pytest.mark.parametrize(("title", "sql"), _panel_queries())
def test_panel_query_runs_against_schema(title: str, sql: str, db_conn: psycopg.Connection) -> None:
    # EXPLAIN validates columns/tables/syntax without depending on data volume.
    db_conn.execute("EXPLAIN " + _strip_macros(sql))
