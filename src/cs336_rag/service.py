"""Application service: one RAG call, timed, with shared clients.

The API holds a single ``RagService`` for the process, so the embedding
and chat clients (and their connection pools) are built once rather than
per request.
"""

import logging
from dataclasses import dataclass
from time import perf_counter

import psycopg
from openai import OpenAI

from cs336_rag.config import Settings
from cs336_rag.embeddings import Embedder, EmbeddingClient
from cs336_rag.llm import build_openai_client
from cs336_rag.rag import RagAnswer, generate_answer, retrieve_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnsweredResult:
    """A RAG answer plus the timings the monitoring dashboards chart."""

    answer: RagAnswer
    retrieval_ms: float
    generation_ms: float

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder or EmbeddingClient(settings)
        self._client = client or build_openai_client(settings, purpose="generate answers")

    def answer(
        self, conn: psycopg.Connection, question: str, variant: str | None = None
    ) -> AnsweredResult:
        """Retrieve context and generate an answer, timing each phase."""
        started = perf_counter()
        chunks = retrieve_context(self._settings, conn, question, embedder=self._embedder)
        retrieved_at = perf_counter()
        answer = generate_answer(
            self._settings,
            question,
            chunks,
            variant=self._settings.rag_prompt_variant if variant is None else variant,
            client=self._client,
        )
        finished = perf_counter()
        return AnsweredResult(
            answer=answer,
            retrieval_ms=(retrieved_at - started) * 1000,
            generation_ms=(finished - retrieved_at) * 1000,
        )
