"""Integration test for the retrieval evaluation runner."""

from unittest.mock import MagicMock

import psycopg
import pytest

from cs336_rag import db
from cs336_rag.config import Settings
from cs336_rag.evals.ground_truth import GroundTruthEntry
from cs336_rag.evals.retrieval_eval import evaluate_retrieval
from tests.conftest import make_chunk

pytestmark = pytest.mark.integration


class OneHotEmbedder:
    """Maps known question texts to one-hot vectors pointing at their chunk."""

    def __init__(self, mapping: dict[str, int], dim: int) -> None:
        self.mapping = mapping
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            vector[self.mapping[text]] = 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def eval_setup(db_conn: psycopg.Connection, db_settings: Settings) -> dict[str, object]:
    chunks = [
        make_chunk(0, "byte pair encoding merges frequent token pairs"),
        make_chunk(1, "attention weighs the relevance of other tokens"),
        make_chunk(2, "GPUs multiply matrices in parallel"),
    ]
    embeddings = [[1.0 if i == j else 0.0 for j in range(8)] for i in range(3)]
    db.replace_chunks(db_conn, chunks, embeddings)

    entries = [
        GroundTruthEntry(question="what does byte pair encoding merge?", chunk_id="vid1:0"),
        GroundTruthEntry(question="what does attention weigh?", chunk_id="vid1:1"),
    ]
    embedder = OneHotEmbedder(
        {entries[0].question: 0, entries[1].question: 1}, dim=db_settings.embedding_dim
    )
    return {"entries": entries, "embedder": embedder}


def test_vector_method_scores_perfectly(
    eval_setup: dict[str, object], db_conn: psycopg.Connection, db_settings: Settings
) -> None:
    report = evaluate_retrieval(
        db_settings,
        db_conn,
        entries=eval_setup["entries"],  # type: ignore[arg-type]
        methods=["vector"],
        embedder=eval_setup["embedder"],  # type: ignore[arg-type]
        limit=5,
    )

    result = report.results["vector"]
    assert result.hit_rate_5 == 1.0
    assert result.mrr == 1.0
    assert result.questions == 2


def test_text_method_finds_keyword_questions(
    eval_setup: dict[str, object], db_conn: psycopg.Connection, db_settings: Settings
) -> None:
    report = evaluate_retrieval(
        db_settings,
        db_conn,
        entries=eval_setup["entries"],  # type: ignore[arg-type]
        methods=["text"],
        embedder=eval_setup["embedder"],  # type: ignore[arg-type]
        limit=5,
    )

    assert report.results["text"].hit_rate_5 == 1.0


def test_hybrid_rerank_uses_injected_http(
    eval_setup: dict[str, object], db_conn: psycopg.Connection, db_settings: Settings
) -> None:
    def fake_post(url: str, json: dict[str, object]) -> MagicMock:
        documents = json["documents"]
        assert isinstance(documents, list)
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(len(documents))
            ]
        }
        return response

    http = MagicMock()
    http.post.side_effect = fake_post

    report = evaluate_retrieval(
        db_settings,
        db_conn,
        entries=eval_setup["entries"],  # type: ignore[arg-type]
        methods=["hybrid_rerank"],
        embedder=eval_setup["embedder"],  # type: ignore[arg-type]
        limit=5,
        rerank_http=http,
    )

    assert http.post.called
    assert report.results["hybrid_rerank"].questions == 2


def test_empty_entries_raise(db_conn: psycopg.Connection, db_settings: Settings) -> None:
    with pytest.raises(ValueError, match="No ground-truth"):
        evaluate_retrieval(db_settings, db_conn, entries=[], methods=["text"], embedder=MagicMock())


def test_stale_ground_truth_raises(
    eval_setup: dict[str, object], db_conn: psycopg.Connection, db_settings: Settings
) -> None:
    stale = [GroundTruthEntry(question="q?", chunk_id="does-not-exist:0")]
    with pytest.raises(ValueError, match="out of sync"):
        evaluate_retrieval(
            db_settings,
            db_conn,
            entries=stale,
            methods=["text"],
            embedder=eval_setup["embedder"],  # type: ignore[arg-type]
        )


def test_report_is_serializable(
    eval_setup: dict[str, object], db_conn: psycopg.Connection, db_settings: Settings
) -> None:
    report = evaluate_retrieval(
        db_settings,
        db_conn,
        entries=eval_setup["entries"],  # type: ignore[arg-type]
        methods=["text", "vector"],
        embedder=eval_setup["embedder"],  # type: ignore[arg-type]
        limit=5,
    )

    dumped = report.model_dump_json()
    assert "hit_rate_5" in dumped
    assert report.limit == 5
