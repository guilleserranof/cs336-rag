"""Shared construction of the OpenAI-compatible API client."""

from openai import OpenAI

from cs336_rag.config import Settings


def build_openai_client(settings: Settings, purpose: str) -> OpenAI:
    """Build a client for the configured endpoint, failing early without a key."""
    if settings.openai_key is None:
        raise ValueError(f"OPENAI_KEY is required to {purpose}")
    return OpenAI(api_key=settings.openai_key, base_url=settings.llm_base_url)
