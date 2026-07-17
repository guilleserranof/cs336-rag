"""Tests for the retrieval layer: text, vector, hybrid RRF, rerank, rewrite."""

from unittest.mock import MagicMock

import psycopg
import pytest

from cs336_rag import db, retrieval
from cs336_rag.models import Chunk
from tests.test_config import make_settings

# ---------------------------------------------------------------------------
# Pure unit tests: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestRrfFuse:
    def test_agreeing_rankings_keep_order(self) -> None:
        fused = retrieval.rrf_fuse([["a", "b", "c"], ["a", "b", "c"]])
        assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]

    def test_item_ranked_high_in_both_beats_single_list_winner(self) -> None:
        fused = retrieval.rrf_fuse([["a", "b"], ["b", "a"]])
        # b: 1/2 + 1/1 vs a: 1/1 + 1/2 (k=0 view) -- with equal totals order
        # falls back to first-seen; use asymmetric case instead
        fused = retrieval.rrf_fuse([["x", "b", "a"], ["b", "a", "y"]])
        assert fused[0][0] == "b"

    def test_item_in_single_ranking_still_included(self) -> None:
        fused = retrieval.rrf_fuse([["a"], ["b"]])
        ids = [chunk_id for chunk_id, _ in fused]
        assert set(ids) == {"a", "b"}

    def test_scores_decrease_with_rank(self) -> None:
        fused = retrieval.rrf_fuse([["a", "b", "c"]])
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_k_softens_rank_differences(self) -> None:
        small_k = retrieval.rrf_fuse([["a", "b"]], k=1)
        large_k = retrieval.rrf_fuse([["a", "b"]], k=1000)
        gap_small = small_k[0][1] - small_k[1][1]
        gap_large = large_k[0][1] - large_k[1][1]
        assert gap_small > gap_large

    def test_empty_input(self) -> None:
        assert retrieval.rrf_fuse([]) == []


# ---------------------------------------------------------------------------
# Unit tests: reranking via the /rerank endpoint (HTTP mocked)
# ---------------------------------------------------------------------------


def make_chunk(index: int, content: str) -> Chunk:
    return Chunk(
        video_id="vid1",
        title="Lecture 1",
        position=1,
        chunk_index=index,
        start=float(index * 10),
        end=float(index * 10 + 10),
        content=content,
    )


class TestRerank:
    def _http_returning(self, results: list[dict[str, float]]) -> MagicMock:
        http = MagicMock()
        response = MagicMock()
        response.json.return_value = {"results": results}
        response.raise_for_status.return_value = None
        http.post.return_value = response
        return http

    def test_reorders_by_relevance_score(self) -> None:
        chunks = [make_chunk(0, "bananas"), make_chunk(1, "attention mechanism")]
        http = self._http_returning(
            [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ]
        )

        reranked = retrieval.rerank_chunks(make_settings(), "what is attention", chunks, http=http)

        assert [result.chunk.chunk_index for result in reranked] == [1, 0]
        assert reranked[0].score == pytest.approx(0.9)

    def test_sends_query_and_documents(self) -> None:
        chunks = [make_chunk(0, "doc zero")]
        http = self._http_returning([{"index": 0, "relevance_score": 0.5}])

        retrieval.rerank_chunks(make_settings(), "q", chunks, http=http)

        payload = http.post.call_args.kwargs["json"]
        assert payload["query"] == "q"
        assert payload["documents"] == ["doc zero"]
        assert payload["model"] == "rerank"

    def test_empty_chunks_short_circuit(self) -> None:
        http = MagicMock()
        assert retrieval.rerank_chunks(make_settings(), "q", [], http=http) == []
        http.post.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests: query rewriting (chat API mocked)
# ---------------------------------------------------------------------------


class TestRewriteQuery:
    def _chat_returning(self, text: str) -> MagicMock:
        client = MagicMock()
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = text
        client.chat.completions.create.return_value = completion
        return client

    def test_returns_rewritten_query(self) -> None:
        client = self._chat_returning("What is the attention mechanism in transformers?")

        rewritten = retrieval.rewrite_query(make_settings(), "whats attn?", client=client)

        assert rewritten == "What is the attention mechanism in transformers?"

    def test_blank_rewrite_falls_back_to_original(self) -> None:
        client = self._chat_returning("   ")
        assert retrieval.rewrite_query(make_settings(), "original", client=client) == "original"

    def test_uses_configured_chat_model(self) -> None:
        client = self._chat_returning("ok")
        retrieval.rewrite_query(make_settings(chat_model="gemma4"), "q", client=client)
        assert client.chat.completions.create.call_args.kwargs["model"] == "gemma4"


# ---------------------------------------------------------------------------
# Integration tests: SQL-backed searches against seeded chunks
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_conn(db_conn: psycopg.Connection) -> psycopg.Connection:
    chunks = [
        make_chunk(0, "the tokenizer splits text into byte pair encoding tokens"),
        make_chunk(1, "attention lets the model focus on relevant context"),
        make_chunk(2, "GPUs execute matrix multiplications in parallel"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    db.replace_chunks(db_conn, chunks, embeddings)
    return db_conn


@pytest.mark.integration
class TestSqlSearches:
    def test_text_search_finds_stemmed_match(self, seeded_conn: psycopg.Connection) -> None:
        results = retrieval.text_search(seeded_conn, "tokenization", limit=2)

        assert results
        assert results[0].chunk.chunk_index == 0
        assert results[0].score > 0

    def test_text_search_no_match_returns_empty(self, seeded_conn: psycopg.Connection) -> None:
        assert retrieval.text_search(seeded_conn, "zebra migration", limit=5) == []

    def test_vector_search_ranks_by_cosine(self, seeded_conn: psycopg.Connection) -> None:
        query_vector = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        results = retrieval.vector_search(seeded_conn, query_vector, limit=3)

        assert [result.chunk.chunk_index for result in results] == [1, 0, 2]
        assert results[0].score > results[1].score

    def test_vector_search_result_carries_chunk_fields(
        self, seeded_conn: psycopg.Connection
    ) -> None:
        result = retrieval.vector_search(seeded_conn, [1.0] + [0.0] * 7, limit=1)[0]

        assert result.chunk.id == "vid1:0"
        assert result.chunk.title == "Lecture 1"
        assert "youtube.com" in result.chunk.url

    def test_hybrid_search_fuses_text_and_vector(self, seeded_conn: psycopg.Connection) -> None:
        # text query matches chunk 0; vector points at chunk 1
        results = retrieval.hybrid_search(
            seeded_conn,
            "byte pair encoding",
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            limit=3,
        )

        top_ids = {result.chunk.chunk_index for result in results[:2]}
        assert top_ids == {0, 1}


class TestSearchDispatch:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            retrieval.search(make_settings(), MagicMock(), "q", method="bm25", embedder=MagicMock())
