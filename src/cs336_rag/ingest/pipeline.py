"""End-to-end ingestion: transcripts on disk -> chunks -> embeddings -> Postgres.

Run with ``uv run cs336-rag ingest``. The pipeline is idempotent: it
atomically replaces the knowledge base on every run, so re-running after a
transcript refresh or a chunking change always yields a consistent state.
"""

import logging
from typing import Protocol

from pydantic import BaseModel

from cs336_rag import db
from cs336_rag.config import Settings
from cs336_rag.ingest.chunking import chunk_transcripts
from cs336_rag.ingest.transcripts import load_transcripts

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns texts into vectors (real API client or test fake)."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class IngestStats(BaseModel):
    videos: int
    chunks: int


def run_ingestion(settings: Settings, embedder: Embedder | None = None) -> IngestStats:
    if embedder is None:
        from cs336_rag.embeddings import EmbeddingClient

        embedder = EmbeddingClient(settings)

    transcripts = load_transcripts(settings.raw_transcripts_dir)
    chunks = chunk_transcripts(transcripts, settings.chunk_max_chars, settings.chunk_overlap_chars)
    logger.info("Chunked %d transcripts into %d chunks", len(transcripts), len(chunks))

    embeddings = embedder.embed([chunk.content for chunk in chunks])
    logger.info("Computed %d embeddings", len(embeddings))

    with db.connect(settings) as conn:
        db.init_schema(conn, settings.embedding_dim)
        inserted = db.replace_chunks(conn, chunks, embeddings)
    logger.info("Loaded %d chunks into Postgres", inserted)

    return IngestStats(videos=len(transcripts), chunks=inserted)
