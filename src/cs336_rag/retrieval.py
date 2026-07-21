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
import re
from typing import Any, get_args

import httpx
import psycopg
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pgvector import Vector
from psycopg.rows import dict_row
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from cs336_rag.config import Settings
from cs336_rag.embeddings import Embedder
from cs336_rag.llm import build_openai_client, build_rerank_http, retry_transient
from cs336_rag.models import Chunk, SearchMethod

logger = logging.getLogger(__name__)

RRF_K = 60  # standard damping constant from the original RRF paper
CANDIDATE_FACTOR = 3  # hybrid fusion / rerank pools consider limit * factor candidates

_CHUNK_COLUMNS = "video_id, title, position, chunk_index, start_s, end_s, content"

REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into clear, self-contained search queries for a "
    "knowledge base of Stanford CS336 'Language Modeling from Scratch' lecture "
    "transcripts. Expand abbreviations and acronyms, fix typos, and drop filler. "
    "Reply with the rewritten query only - no quotes, no explanations."
)


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


def _row_to_result(row: dict[str, Any], score: float) -> SearchResult:
    chunk = Chunk(
        video_id=row["video_id"],
        title=row["title"],
        position=row["position"],
        chunk_index=row["chunk_index"],
        start=row["start_s"],
        end=row["end_s"],
        content=row["content"],
    )
    return SearchResult(chunk=chunk, score=score)


def _or_tsquery(query: str) -> str:
    """Build an OR tsquery string from free text.

    ``websearch_to_tsquery`` ANDs every term, so a full-sentence question
    only matches a chunk that literally contains *all* of its words —
    almost never true for paraphrased natural-language queries. ORing the
    terms lets any overlap match, and ``ts_rank_cd`` then ranks by how many
    query terms are covered and how densely. Terms are reduced to bare
    word characters (no tsquery operators survive), so the result is always
    a safe ``to_tsquery`` input; stopwords are dropped by the ``english``
    config.
    """
    terms = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
    return " | ".join(terms)


def text_search(conn: psycopg.Connection, query: str, limit: int = 10) -> list[SearchResult]:
    """Full-text search with English stemming, ranked by cover density."""
    tsquery = _or_tsquery(query)
    if not tsquery:
        return []
    with conn.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(
            f"""
            SELECT {_CHUNK_COLUMNS}, ts_rank_cd(tsv, query) AS score
            FROM chunks, to_tsquery('english', %s) AS query
            WHERE tsv @@ query
            ORDER BY score DESC
            LIMIT %s
            """,
            (tsquery, limit),
        ).fetchall()
    return [_row_to_result(row, float(row["score"])) for row in rows]


def vector_search(
    conn: psycopg.Connection, query_vector: list[float], limit: int = 10
) -> list[SearchResult]:
    """Nearest neighbours by cosine distance, reported as similarity.

    The ORDER BY must be the bare ``<=>`` operator ascending — any wrapping
    arithmetic (or DESC) makes the HNSW index unusable and falls back to a
    sequential scan.
    """
    with conn.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(
            f"""
            SELECT {_CHUNK_COLUMNS}, embedding <=> %s AS distance
            FROM chunks
            ORDER BY distance
            LIMIT %s
            """,
            (Vector(query_vector), limit),
        ).fetchall()
    return [_row_to_result(row, 1.0 - float(row["distance"])) for row in rows]


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
    candidates: int | None = None,
) -> list[SearchResult]:
    """Fuse text and vector rankings with RRF and return the top ``limit``.

    ``candidates`` sets how many rows each source contributes to the fusion
    (defaults to ``limit * CANDIDATE_FACTOR``).
    """
    per_source = candidates if candidates is not None else limit * CANDIDATE_FACTOR
    text_results = text_search(conn, query, limit=per_source)
    vector_results = vector_search(conn, query_vector, limit=per_source)

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


def _retryable_http_error(error: BaseException) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code >= 500 or error.response.status_code == 429
    )


@retry(
    retry=retry_if_exception(_retryable_http_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=30),
    reraise=True,
)
def _post_rerank(
    http: httpx.Client, settings: Settings, query: str, documents: list[str]
) -> list[dict[str, Any]]:
    response = http.post(
        "/rerank",
        json={
            "model": settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        },
    )
    response.raise_for_status()
    results: list[dict[str, Any]] = response.json()["results"]
    return results


def rerank_chunks(
    settings: Settings,
    query: str,
    chunks: list[Chunk],
    http: httpx.Client | None = None,
) -> list[SearchResult]:
    """Re-score chunks with the cross-encoder ``/rerank`` endpoint."""
    if not chunks:
        return []
    if http is None:
        with build_rerank_http(settings) as owned_http:
            return rerank_chunks(settings, query, chunks, http=owned_http)

    results = _post_rerank(http, settings, query, [chunk.content for chunk in chunks])
    valid = [item for item in results if 0 <= item["index"] < len(chunks)]
    if len(valid) < len(results):
        logger.warning(
            "Rerank returned %d out-of-range indices; ignoring them", len(results) - len(valid)
        )
    ordered = sorted(valid, key=lambda item: item["relevance_score"], reverse=True)
    return [
        SearchResult(chunk=chunks[item["index"]], score=float(item["relevance_score"]))
        for item in ordered
    ]


@retry_transient
def _create_rewrite(client: OpenAI, settings: Settings, query: str) -> ChatCompletion:
    return client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=120,
    )


def rewrite_query(settings: Settings, query: str, client: OpenAI | None = None) -> str:
    """Rewrite a user question into a cleaner search query via the chat model.

    Falls back to the original query if the model returns nothing usable.
    """
    client = client or build_openai_client(settings, purpose="rewrite queries")
    completion = _create_rewrite(client, settings, query)
    if not completion.choices:
        logger.warning("Query rewrite returned no choices; keeping original query")
        return query
    rewritten = (completion.choices[0].message.content or "").strip()
    if not rewritten:
        logger.warning("Query rewrite returned empty output; keeping original query")
        return query
    return rewritten


def search(
    settings: Settings,
    conn: psycopg.Connection,
    query: str,
    method: SearchMethod | None = None,
    limit: int = 10,
    embedder: Embedder | None = None,
    *,
    query_vector: list[float] | None = None,
    rerank_http: httpx.Client | None = None,
) -> list[SearchResult]:
    """Dispatch to a search method by name (the evaluation compares them all).

    ``method`` defaults to ``settings.retrieval_method`` (the evaluation
    winner; see docs/evaluation.md). Callers that already hold a question
    embedding or a shared rerank client (the evaluation runner) can pass
    ``query_vector`` / ``rerank_http`` so the served pipeline and the
    evaluated pipeline are the same code path.
    """
    method = settings.retrieval_method if method is None else method
    if method not in get_args(SearchMethod):  # guard untyped callers (CLI args, eval configs)
        raise ValueError(f"Unknown search method {method!r}; expected {get_args(SearchMethod)}")

    if method == "text":
        return text_search(conn, query, limit)

    if query_vector is None:
        if embedder is None:
            from cs336_rag.embeddings import EmbeddingClient

            embedder = EmbeddingClient(settings)
        query_vector = embedder.embed([query])[0]

    if method == "vector":
        return vector_search(conn, query_vector, limit)
    if method == "hybrid":
        return hybrid_search(conn, query, query_vector, limit)

    # hybrid_rerank: build a wider candidate pool, let the cross-encoder pick
    pool = limit * CANDIDATE_FACTOR
    candidates = hybrid_search(conn, query, query_vector, limit=pool, candidates=pool)
    reranked = rerank_chunks(
        settings, query, [result.chunk for result in candidates], http=rerank_http
    )
    return reranked[:limit]
