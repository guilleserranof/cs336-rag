"""Integration tests for the Postgres/pgvector layer."""

import psycopg
import pytest

from cs336_rag import db
from cs336_rag.config import Settings
from cs336_rag.models import Chunk

pytestmark = pytest.mark.integration


def make_chunk(index: int = 0, content: str = "attention is all you need") -> Chunk:
    return Chunk(
        video_id="vid1",
        title="Lecture 1",
        position=1,
        chunk_index=index,
        start=float(index * 10),
        end=float(index * 10 + 10),
        content=content,
    )


def test_init_schema_is_idempotent(db_conn: psycopg.Connection, db_settings: Settings) -> None:
    db.init_schema(db_conn, db_settings.embedding_dim)
    db.init_schema(db_conn, db_settings.embedding_dim)

    row = db_conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert row is not None


def test_replace_chunks_inserts_rows(db_conn: psycopg.Connection) -> None:
    chunks = [make_chunk(0), make_chunk(1, "tokenization splits text")]
    embeddings = [[0.1] * 8, [0.2] * 8]

    inserted = db.replace_chunks(db_conn, chunks, embeddings)

    assert inserted == 2
    assert db.count_chunks(db_conn) == 2


def test_replace_chunks_replaces_previous_content(db_conn: psycopg.Connection) -> None:
    db.replace_chunks(db_conn, [make_chunk(0)], [[0.1] * 8])
    db.replace_chunks(db_conn, [make_chunk(1, "new content")], [[0.3] * 8])

    assert db.count_chunks(db_conn) == 1
    row = db_conn.execute("SELECT content FROM chunks").fetchone()
    assert row is not None
    assert row[0] == "new content"


def test_chunk_row_stores_metadata_and_tsvector(db_conn: psycopg.Connection) -> None:
    db.replace_chunks(db_conn, [make_chunk(3, "the transformer architecture")], [[0.5] * 8])

    row = db_conn.execute(
        "SELECT id, video_id, title, position, start_s, end_s FROM chunks"
    ).fetchone()
    assert row == ("vid1:3", "vid1", "Lecture 1", 1, 30.0, 40.0)

    matches = db_conn.execute(
        "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', 'transformers')"
    ).fetchone()
    assert matches is not None
    assert matches[0] == 1  # stemming: 'transformers' matches 'transformer'


def test_embedding_roundtrip(db_conn: psycopg.Connection) -> None:
    vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    db.replace_chunks(db_conn, [make_chunk(0)], [vector])

    row = db_conn.execute("SELECT embedding FROM chunks").fetchone()
    assert row is not None
    stored = row[0].to_list()
    assert stored == pytest.approx(vector)


def test_mismatched_lengths_raise(db_conn: psycopg.Connection) -> None:
    with pytest.raises(ValueError, match="length"):
        db.replace_chunks(db_conn, [make_chunk(0)], [])


def test_empty_chunks_never_wipe_the_knowledge_base(db_conn: psycopg.Connection) -> None:
    db.replace_chunks(db_conn, [make_chunk(0)], [[0.1] * 8])

    with pytest.raises(ValueError, match="zero chunks"):
        db.replace_chunks(db_conn, [], [])

    assert db.count_chunks(db_conn) == 1  # previous content untouched


def test_init_schema_recreates_table_on_dimension_change(
    db_conn: psycopg.Connection, db_settings: "Settings"
) -> None:
    db.replace_chunks(db_conn, [make_chunk(0)], [[0.1] * 8])

    db.init_schema(db_conn, embedding_dim=4)

    row = db_conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    ).fetchone()
    assert row == ("vector(4)",)
    assert db.count_chunks(db_conn) == 0  # rebuilt empty, ready for re-ingest

    # restore the fixture's dimension for subsequent tests
    db.init_schema(db_conn, db_settings.embedding_dim)
