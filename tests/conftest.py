"""Shared fixtures. Integration tests need a running Postgres with pgvector.

Locally: ``docker compose up -d postgres``. In CI the pgvector service is
provided by the workflow. Without a reachable server the integration tests
are skipped.
"""

from collections.abc import Iterator

import psycopg
import pytest

from cs336_rag.config import Settings
from tests.test_config import make_settings

TEST_DB = "cs336_rag_test"
EMBEDDING_DIM = 8  # keep test vectors tiny


@pytest.fixture(scope="session")
def db_settings() -> Settings:
    """Settings pointing at a dedicated test database (created on demand)."""
    base = Settings(_env_file=None, openai_key=None)  # type: ignore[call-arg]
    settings = make_settings(
        db_host=base.db_host,
        db_port=base.db_port,
        db_user=base.db_user,
        db_password=base.db_password,
        db_name=TEST_DB,
        embedding_dim=EMBEDDING_DIM,
    )
    admin_dsn = base.db_dsn.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    except psycopg.OperationalError:
        pytest.skip("Postgres is not reachable; run 'docker compose up -d postgres'")
    return settings


@pytest.fixture
def db_conn(db_settings: Settings) -> Iterator[psycopg.Connection]:
    """A clean, schema-initialised connection per test."""
    from cs336_rag import db

    with db.connect(db_settings) as conn:
        db.init_schema(conn, db_settings.embedding_dim)
        conn.execute("TRUNCATE chunks")
        conn.commit()
        yield conn
