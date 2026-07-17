"""Thin wrapper around the OpenAI-compatible embeddings endpoint.

Adds batching (the API rejects oversized requests), retry with exponential
backoff on transient failures, and the ``dimensions`` parameter
(qwen3-embedding supports matryoshka truncation; 1024 dims keeps
pgvector's HNSW index applicable and returns unit-norm vectors).
"""

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cs336_rag.config import Settings
from cs336_rag.llm import build_openai_client

# transient failures worth retrying; auth/validation errors fail fast
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class EmbeddingClient:
    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self._client = client or build_openai_client(settings, purpose="compute embeddings")
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dim
        self._batch_size = settings.embed_batch_size

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=30),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model, input=texts, dimensions=self._dimensions
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in input order, batching requests."""
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[offset : offset + self._batch_size]))
        return vectors
