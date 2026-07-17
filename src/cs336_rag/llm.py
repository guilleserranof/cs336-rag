"""Shared construction of clients for the OpenAI-compatible API.

All auth/base-url wiring lives here so the embedding, chat and rerank
paths cannot drift apart, together with the house retry policy for
transient API failures.
"""

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cs336_rag.config import Settings

# transient failures worth retrying; auth/validation errors fail fast
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)

retry_transient = retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=30),
    reraise=True,
)


def build_openai_client(settings: Settings, purpose: str) -> OpenAI:
    """Build a client for the configured endpoint, failing early without a key."""
    if settings.openai_key is None:
        raise ValueError(f"OPENAI_KEY is required to {purpose}")
    return OpenAI(api_key=settings.openai_key, base_url=settings.llm_base_url)


def build_rerank_http(settings: Settings, purpose: str = "rerank documents") -> httpx.Client:
    """HTTP client for API routes the OpenAI SDK does not cover (``/rerank``)."""
    if settings.openai_key is None:
        raise ValueError(f"OPENAI_KEY is required to {purpose}")
    return httpx.Client(
        base_url=settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.openai_key}"},
        timeout=30.0,
    )
