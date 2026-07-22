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
from cs336_rag.models import Chunk
from cs336_rag.rag import RagAnswer, generate_answer, retrieve_context
from cs336_rag.retrieval import rewrite_query

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

    def retrieve(self, conn: psycopg.Connection, question: str) -> tuple[list[Chunk], float]:
        """Fetch context chunks. The only phase that needs the database.

        With ``rag_rewrite_query`` enabled the question is first rewritten into
        a cleaner search query; the original question still drives the answer
        prompt (see ``generate``).
        """
        started = perf_counter()
        search_query = question
        if self._settings.rag_rewrite_query:
            search_query = rewrite_query(self._settings, question, client=self._client)
        chunks = retrieve_context(self._settings, conn, search_query, embedder=self._embedder)
        return chunks, (perf_counter() - started) * 1000

    def generate(
        self, question: str, chunks: list[Chunk], variant: str | None = None
    ) -> tuple[RagAnswer, float]:
        """Generate the answer. Deliberately takes no connection: this call
        can run for a minute and must not hold a pooled connection open."""
        started = perf_counter()
        answer = generate_answer(
            self._settings,
            question,
            chunks,
            variant=self._settings.rag_prompt_variant if variant is None else variant,
            client=self._client,
        )
        return answer, (perf_counter() - started) * 1000

    def answer(
        self, conn: psycopg.Connection, question: str, variant: str | None = None
    ) -> AnsweredResult:
        """Both phases at once, for callers that already hold a connection."""
        chunks, retrieval_ms = self.retrieve(conn, question)
        answer, generation_ms = self.generate(question, chunks, variant)
        return AnsweredResult(answer=answer, retrieval_ms=retrieval_ms, generation_ms=generation_ms)
