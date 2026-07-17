"""Compare retrieval methods on the ground-truth question set.

For every question we know which chunk it was generated from; each method
is scored by hit rate@5/@10 and MRR of that chunk in its ranking.
Question embeddings are computed once and shared across the vector-based
methods, and the rerank HTTP connection is reused across questions.
"""

import logging
from datetime import UTC, datetime

import httpx
import psycopg
from pydantic import BaseModel

from cs336_rag import retrieval
from cs336_rag.config import Settings
from cs336_rag.embeddings import Embedder
from cs336_rag.evals.ground_truth import GroundTruthEntry
from cs336_rag.evals.metrics import hit_rate_at, mrr, rank_of
from cs336_rag.llm import build_rerank_http
from cs336_rag.retrieval import SearchMethod

logger = logging.getLogger(__name__)


class MethodResult(BaseModel):
    hit_rate_5: float
    hit_rate_10: float
    mrr: float
    questions: int


class RetrievalReport(BaseModel):
    generated_at: datetime
    limit: int
    results: dict[str, MethodResult]

    def as_markdown(self) -> str:
        lines = [
            "| Method | Hit rate@5 | Hit rate@10 | MRR |",
            "|---|---|---|---|",
        ]
        lines.extend(
            f"| {method} | {result.hit_rate_5:.3f} | {result.hit_rate_10:.3f} | {result.mrr:.3f} |"
            for method, result in self.results.items()
        )
        return "\n".join(lines)


def _retrieved_ids(
    settings: Settings,
    conn: psycopg.Connection,
    method: SearchMethod,
    entry: GroundTruthEntry,
    question_vector: list[float] | None,
    limit: int,
    rerank_http: httpx.Client | None,
) -> list[str]:
    if method == "text":
        results = retrieval.text_search(conn, entry.question, limit)
    elif method == "vector":
        assert question_vector is not None
        results = retrieval.vector_search(conn, question_vector, limit)
    elif method == "hybrid":
        assert question_vector is not None
        results = retrieval.hybrid_search(conn, entry.question, question_vector, limit)
    else:  # hybrid_rerank
        assert question_vector is not None
        pool = limit * retrieval.CANDIDATE_FACTOR
        candidates = retrieval.hybrid_search(
            conn, entry.question, question_vector, limit=pool, candidates=pool
        )
        reranked = retrieval.rerank_chunks(
            settings, entry.question, [result.chunk for result in candidates], http=rerank_http
        )
        results = reranked[:limit]
    return [result.chunk.id for result in results]


def evaluate_retrieval(
    settings: Settings,
    conn: psycopg.Connection,
    entries: list[GroundTruthEntry],
    methods: list[SearchMethod],
    embedder: Embedder,
    limit: int = 10,
    rerank_http: httpx.Client | None = None,
) -> RetrievalReport:
    """Score every method on the full question set."""
    needs_vectors = any(method != "text" for method in methods)
    question_vectors: list[list[float] | None]
    if needs_vectors:
        question_vectors = list(embedder.embed([entry.question for entry in entries]))
    else:
        question_vectors = [None] * len(entries)

    needs_rerank = "hybrid_rerank" in methods
    owned_http = None
    if needs_rerank and rerank_http is None:
        owned_http = rerank_http = build_rerank_http(settings)

    try:
        results: dict[str, MethodResult] = {}
        for method in methods:
            ranks: list[int | None] = []
            for entry, question_vector in zip(entries, question_vectors, strict=True):
                retrieved = _retrieved_ids(
                    settings, conn, method, entry, question_vector, limit, rerank_http
                )
                ranks.append(rank_of(retrieved, entry.chunk_id))
            results[method] = MethodResult(
                hit_rate_5=hit_rate_at(ranks, 5),
                hit_rate_10=hit_rate_at(ranks, 10),
                mrr=mrr(ranks),
                questions=len(ranks),
            )
            logger.info(
                "%s: hit@5=%.3f hit@10=%.3f mrr=%.3f",
                method,
                results[method].hit_rate_5,
                results[method].hit_rate_10,
                results[method].mrr,
            )
    finally:
        if owned_http is not None:
            owned_http.close()

    return RetrievalReport(generated_at=datetime.now(UTC), limit=limit, results=results)
