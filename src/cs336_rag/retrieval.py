"""Retrieval over the pgvector knowledge base.

Four search methods, all returning ranked ``SearchResult`` lists:

- ``text_search``   — Postgres full-text search (GIN over a generated
  tsvector), ranked with ``ts_rank_cd``.
- ``vector_search`` — pgvector cosine similarity over qwen3 embeddings
  (HNSW index).
- ``hybrid_search`` — Reciprocal Rank Fusion of the two lists above.
- ``hybrid + rerank`` — hybrid candidates re-scored by the ``rerank``
  cross-encoder endpoint.

``rewrite_query`` optionally normalises a user question with the chat
model before searching (expands acronyms, strips filler).

The relative quality of these methods is measured in the retrieval
evaluation (see ``cs336_rag.evals``); the app serves the winner.
"""

import logging
from typing import Any, Literal, get_args

import httpx
import psycopg
from openai import OpenAI
from pgvector import Vector
from pydantic import BaseModel

from cs336_rag.config import Settings
from cs336_rag.ingest.pipeline import Embedder
from cs336_rag.llm import build_openai_client
from cs336_rag.models import Chunk

logger = logging.getLogger(__name__)

SearchMethod = Literal["text", "vector", "hybrid", "hybrid_rerank"]

RRF_K = 60  # standard damping constant from the original RRF paper
CANDIDATE_FACTOR = 3  # hybrid/rerank consider limit * factor candidates per source

_CHUNK_COLUMNS = "id, video_id, title, position, chunk_index, start_s, end_s, content"

REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into clear, self-contained search queries for a "
    "knowledge base of Stanford CS336 'Language Modeling from Scratch' lecture "
    "transcripts. Expand abbreviations and acronyms, fix typos, and drop filler. "
    "Reply with the rewritten query only - no quotes, no explanations."
)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


def _row_to_result(row: tuple[Any, ...]) -> SearchResult:
    (_, video_id, title, position, chunk_index, start_s, end_s, content, score) = row
    chunk = Chunk(
        video_id=video_id,
        title=title,
        position=position,
        chunk_index=chunk_index,
        start=start_s,
        end=end_s,
        content=content,
    )
    return SearchResult(chunk=chunk, score=float(score))


def text_search(conn: psycopg.Connection, query: str, limit: int = 10) -> list[SearchResult]:
    """Full-text search with English stemming, ranked by cover density."""
    rows = conn.execute(
        f"""
        SELECT {_CHUNK_COLUMNS}, ts_rank_cd(tsv, query) AS score
        FROM chunks, websearch_to_tsquery('english', %s) AS query
        WHERE tsv @@ query
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, limit),
    ).fetchall()
    return [_row_to_result(row) for row in rows]


def vector_search(
    conn: psycopg.Connection, query_vector: list[float], limit: int = 10
) -> list[SearchResult]:
    """Nearest neighbours by cosine similarity (1 - cosine distance)."""
    rows = conn.execute(
        f"""
        SELECT {_CHUNK_COLUMNS}, 1 - (embedding <=> %s) AS score
        FROM chunks
        ORDER BY score DESC
        LIMIT %s
        """,
        (Vector(query_vector), limit),
    ).fetchall()
    return [_row_to_result(row) for row in rows]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(id) = sum over rankings of 1/(k + rank).

    Returns (id, fused_score) pairs sorted by descending score. Stable for
    ties: first-seen ids win.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(
    conn: psycopg.Connection,
    query: str,
    query_vector: list[float],
    limit: int = 10,
    k: int = RRF_K,
) -> list[SearchResult]:
    """Fuse text and vector rankings with RRF and return the top ``limit``."""
    candidates = limit * CANDIDATE_FACTOR
    text_results = text_search(conn, query, limit=candidates)
    vector_results = vector_search(conn, query_vector, limit=candidates)

    by_id = {result.chunk.id: result for result in [*text_results, *vector_results]}
    fused = rrf_fuse(
        [
            [result.chunk.id for result in text_results],
            [result.chunk.id for result in vector_results],
        ],
        k=k,
    )
    return [
        SearchResult(chunk=by_id[chunk_id].chunk, score=score) for chunk_id, score in fused[:limit]
    ]


def rerank_chunks(
    settings: Settings,
    query: str,
    chunks: list[Chunk],
    top_n: int | None = None,
    http: httpx.Client | None = None,
) -> list[SearchResult]:
    """Re-score chunks with the cross-encoder ``/rerank`` endpoint."""
    if not chunks:
        return []
    if http is None:
        if settings.openai_key is None:
            raise ValueError("OPENAI_KEY is required to rerank documents")
        http = httpx.Client(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.openai_key}"},
            timeout=30.0,
        )
    response = http.post(
        "/rerank",
        json={
            "model": settings.rerank_model,
            "query": query,
            "documents": [chunk.content for chunk in chunks],
            "top_n": top_n or len(chunks),
        },
    )
    response.raise_for_status()
    results = response.json()["results"]
    ordered = sorted(results, key=lambda item: item["relevance_score"], reverse=True)
    return [
        SearchResult(chunk=chunks[item["index"]], score=float(item["relevance_score"]))
        for item in ordered
    ]


def rewrite_query(settings: Settings, query: str, client: OpenAI | None = None) -> str:
    """Rewrite a user question into a cleaner search query via the chat model.

    Falls back to the original query if the model returns nothing useful.
    """
    client = client or build_openai_client(settings, purpose="rewrite queries")
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    rewritten = (completion.choices[0].message.content or "").strip()
    if not rewritten:
        logger.warning("Query rewrite returned empty output; keeping original query")
        return query
    return rewritten


def search(
    settings: Settings,
    conn: psycopg.Connection,
    query: str,
    method: str = "hybrid_rerank",
    limit: int = 10,
    embedder: Embedder | None = None,
) -> list[SearchResult]:
    """Dispatch to a search method by name (the evaluation compares them all)."""
    if method not in get_args(SearchMethod):
        raise ValueError(f"Unknown search method {method!r}; expected {get_args(SearchMethod)}")

    if method == "text":
        return text_search(conn, query, limit)

    if embedder is None:
        from cs336_rag.embeddings import EmbeddingClient

        embedder = EmbeddingClient(settings)
    query_vector = embedder.embed([query])[0]

    if method == "vector":
        return vector_search(conn, query_vector, limit)
    if method == "hybrid":
        return hybrid_search(conn, query, query_vector, limit)

    # hybrid_rerank: over-fetch hybrid candidates, let the cross-encoder pick
    candidates = hybrid_search(conn, query, query_vector, limit * CANDIDATE_FACTOR)
    reranked = rerank_chunks(settings, query, [result.chunk for result in candidates])
    return reranked[:limit]
