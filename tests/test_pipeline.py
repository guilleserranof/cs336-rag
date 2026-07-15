"""Integration test for the full ingestion pipeline (with a fake embedder)."""

from pathlib import Path

import psycopg
import pytest

from cs336_rag import db
from cs336_rag.config import Settings
from cs336_rag.ingest.pipeline import run_ingestion
from cs336_rag.ingest.transcripts import save_transcript
from cs336_rag.models import TranscriptSegment, VideoTranscript

pytestmark = pytest.mark.integration


class FakeEmbedder:
    """Deterministic stand-in for the embedding API."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(i % 7) / 7] * self.dim for i, _ in enumerate(texts)]


@pytest.fixture
def transcripts_on_disk(db_settings: Settings, tmp_path: Path) -> Settings:
    settings = db_settings.model_copy(update={"data_dir": tmp_path})
    for position in (1, 2):
        transcript = VideoTranscript(
            video_id=f"vid{position}",
            title=f"Lecture {position}",
            position=position,
            source="youtube",
            segments=[
                TranscriptSegment(
                    text=f"sentence {i} of lecture {position}", start=i * 3, duration=3.0
                )
                for i in range(40)
            ],
        )
        save_transcript(transcript, settings.raw_transcripts_dir)
    return settings


def test_pipeline_ingests_all_transcripts(
    transcripts_on_disk: Settings, db_conn: psycopg.Connection
) -> None:
    settings = transcripts_on_disk
    embedder = FakeEmbedder(settings.embedding_dim)

    stats = run_ingestion(settings, embedder=embedder)

    assert stats.videos == 2
    assert stats.chunks > 2
    with db.connect(settings) as conn:
        assert db.count_chunks(conn) == stats.chunks
        videos = conn.execute("SELECT DISTINCT video_id FROM chunks ORDER BY video_id").fetchall()
        assert videos == [("vid1",), ("vid2",)]


def test_pipeline_is_idempotent(
    transcripts_on_disk: Settings, db_conn: psycopg.Connection
) -> None:
    settings = transcripts_on_disk
    embedder = FakeEmbedder(settings.embedding_dim)

    first = run_ingestion(settings, embedder=embedder)
    second = run_ingestion(settings, embedder=embedder)

    assert first.chunks == second.chunks
    with db.connect(settings) as conn:
        assert db.count_chunks(conn) == first.chunks
