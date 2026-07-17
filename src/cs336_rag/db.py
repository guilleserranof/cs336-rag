"""Postgres/pgvector access layer."""

import logging
from importlib import resources

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from cs336_rag.config import Settings
from cs336_rag.models import Chunk

logger = logging.getLogger(__name__)


def connect(settings: Settings) -> psycopg.Connection:
    """Open a connection with the pgvector type adapter registered.

    On a brand-new database the vector extension does not exist yet, so
    registration fails once; we bootstrap the extension and retry. After
    that first connection, opening a connection issues no DDL.
    """
    conn = psycopg.connect(settings.db_dsn)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
    return conn


def _current_embedding_dim(conn: psycopg.Connection) -> int | None:
    """Dimension of chunks.embedding, or None if the table does not exist."""
    row = conn.execute(
        """
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'
        """
    ).fetchone()
    return int(row[0]) if row else None


def init_schema(conn: psycopg.Connection, embedding_dim: int) -> None:
    """Create tables and indexes; safe to run repeatedly.

    If the configured embedding dimension differs from the existing
    column, the chunks table is dropped and recreated: every ingestion
    run rebuilds its full content anyway, and without this the stale
    dimension would reject all inserts.
    """
    current = _current_embedding_dim(conn)
    if current is not None and current != embedding_dim:
        logger.warning(
            "Embedding dimension changed (%d -> %d); recreating the chunks table",
            current,
            embedding_dim,
        )
        conn.execute("DROP TABLE chunks")
    template = resources.files("cs336_rag").joinpath("schema.sql").read_text(encoding="utf-8")
    # targeted replace instead of str.format: SQL is full of braces-in-waiting
    conn.execute(template.replace("{embedding_dim}", str(int(embedding_dim))))
    conn.commit()


def replace_chunks(
    conn: psycopg.Connection, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Atomically replace the knowledge base content with the given chunks."""
    if not chunks:
        raise ValueError("Refusing to replace the knowledge base with zero chunks")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
        )
    with conn.cursor() as cursor:
        cursor.execute("TRUNCATE chunks")
        cursor.executemany(
            """
            INSERT INTO chunks
                (id, video_id, title, position, chunk_index, start_s, end_s, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk.id,
                    chunk.video_id,
                    chunk.title,
                    chunk.position,
                    chunk.chunk_index,
                    chunk.start,
                    chunk.end,
                    chunk.content,
                    Vector(embedding),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
    conn.commit()
    return len(chunks)


def count_chunks(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert row is not None
    return int(row[0])
