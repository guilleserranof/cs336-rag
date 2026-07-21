"""Shared fixtures. Integration tests need a running Postgres with pgvector.

Locally: ``docker compose up -d postgres``. In CI the pgvector service is
provided by the workflow. Without a reachable server the integration tests
are skipped.
"""

import os
from collections.abc import Iterator

import psycopg
import pytest

from cs336_rag.config import Settings
from cs336_rag.models import Chunk

TEST_DB = "cs336_rag_test"
EMBEDDING_DIM = 8  # keep test vectors tiny


def make_chunk(index: int = 0, content: str = "attention is all you need") -> Chunk:
    """Shared Chunk factory for db/retrieval tests."""
    return Chunk(
        video_id="vid1",
        title="Lecture 1",
        position=1,
        chunk_index=index,
        start=float(index * 10),
        end=float(index * 10 + 10),
        content=content,
    )


@pytest.fixture(scope="session")
def db_settings() -> Settings:
    """Settings pointing at a dedicated test database (created on demand)."""
    base = Settings(_env_file=None, openai_key=None)  # type: ignore[call-arg]
    settings = base.model_copy(update={"db_name": TEST_DB, "embedding_dim": EMBEDDING_DIM})
    admin_dsn = base.db_dsn.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    except psycopg.OperationalError as error:
        message = "Postgres is not reachable; run 'docker compose up -d postgres'"
        if os.getenv("CI") == "true":
            pytest.fail(f"{message}. CI must run integration tests: {error}", pytrace=False)
        pytest.skip(message)
    return settings


@pytest.fixture
def db_conn(db_settings: Settings) -> Iterator[psycopg.Connection]:
    """A clean, schema-initialised connection per test."""
    from cs336_rag import db

    with db.connect(db_settings) as conn:
        db.init_schema(conn, db_settings.embedding_dim)
        # feedback cascades from conversations
        conn.execute("TRUNCATE chunks, conversations CASCADE")
        conn.commit()
        yield conn
