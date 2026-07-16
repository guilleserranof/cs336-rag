"""Postgres/pgvector access layer."""

from importlib import resources

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from cs336_rag.config import Settings
from cs336_rag.models import Chunk


def connect(settings: Settings) -> psycopg.Connection:
    """Open a connection with the pgvector type adapter registered.

    The extension must exist before ``register_vector`` works, so it is
    created here (idempotent, cheap) rather than only in ``init_schema``.
    """
    conn = psycopg.connect(settings.db_dsn)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, embedding_dim: int) -> None:
    """Create tables and indexes; safe to run repeatedly."""
    template = resources.files("cs336_rag").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.execute(template.format(embedding_dim=int(embedding_dim)))
    conn.commit()


def replace_chunks(
    conn: psycopg.Connection, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Atomically replace the knowledge base content with the given chunks."""
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
