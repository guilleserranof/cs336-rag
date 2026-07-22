"""Retrieval-Augmented Generation answer flow.

Retrieves the most relevant lecture chunks (via the evaluation-winning
retrieval method) and asks the chat model to answer grounded in them,
citing sources by number. Several prompt variants are defined and compared
in the answer evaluation (``cs336_rag.evals.answer_eval``); the app serves
the winner (``Settings.rag_prompt_variant``).
"""

import logging
from typing import Any

import psycopg
from openai import OpenAI
from pydantic import BaseModel

from cs336_rag import retrieval
from cs336_rag.config import Settings
from cs336_rag.embeddings import Embedder
from cs336_rag.llm import build_openai_client, retry_transient, thinking_extra_body
from cs336_rag.models import Chunk

logger = logging.getLogger(__name__)


class EmptyAnswerError(RuntimeError):
    """The model returned no answer text (filtered, truncated or empty)."""


class PromptVariant(BaseModel):
    """A system prompt paired with generation parameters, keyed by name in
    ``PROMPT_VARIANTS``."""

    system: str
    temperature: float = 0.2


# Three deliberately different answering strategies, scored in the answer eval.
PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "baseline": PromptVariant(
        system=(
            "You answer questions about the Stanford CS336 lecture series using the "
            "provided context passages. Answer the question."
        ),
    ),
    "grounded": PromptVariant(
        system=(
            "You are a teaching assistant for Stanford CS336 'Language Modeling from "
            "Scratch'. Answer the question using ONLY the numbered context passages "
            "below. Cite the passages you use inline as [1], [2], etc. If the context "
            "does not contain the answer, say so plainly instead of guessing. Be "
            "accurate and specific."
        ),
    ),
    "tutor": PromptVariant(
        system=(
            "You are an encouraging teaching assistant for Stanford CS336. Using only "
            "the numbered context passages, explain the answer clearly enough for a "
            "student new to the topic: define key terms and give the intuition, then "
            "the specifics. Cite passages inline as [1], [2]. If the context lacks the "
            "answer, say what is missing rather than inventing it."
        ),
    ),
}


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class RagAnswer(BaseModel):
    question: str
    answer: str
    variant: str
    sources: list[Chunk]
    usage: TokenUsage | None = None


def format_context(chunks: list[Chunk]) -> str:
    """Render chunks as a numbered source list for the prompt."""
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{index}] {chunk.title} ({chunk.url})\n{chunk.content}")
    return "\n\n".join(blocks)


def build_messages(variant: str, question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    """Build the chat messages for a prompt variant."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant {variant!r}; expected {list(PROMPT_VARIANTS)}")
    context = format_context(chunks) if chunks else "(no context retrieved)"
    user = f"Context passages:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": PROMPT_VARIANTS[variant].system},
        {"role": "user", "content": user},
    ]


def _extract_usage(completion: object) -> TokenUsage | None:
    """Read token counts when the backend reports them (best effort)."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    try:
        return TokenUsage(
            prompt_tokens=int(usage.prompt_tokens),
            completion_tokens=int(usage.completion_tokens),
        )
    except (AttributeError, TypeError, ValueError):
        return None


@retry_transient
def _create_answer(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    extra_body: dict[str, Any] | None = None,
) -> tuple[str, TokenUsage | None]:
    # extra_body=None is a no-op, so backends that do not understand the
    # thinking kwarg still receive a plain request
    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        extra_body=extra_body,
    )
    usage = _extract_usage(completion)
    if not completion.choices:
        return "", usage
    return (completion.choices[0].message.content or "").strip(), usage


def generate_answer(
    settings: Settings,
    question: str,
    chunks: list[Chunk],
    variant: str,
    client: OpenAI | None = None,
) -> RagAnswer:
    """Generate an answer for a question from already-retrieved chunks.

    Raises ``EmptyAnswerError`` when the model returns nothing, so callers can
    tell a generation failure from a genuinely poor answer.
    """
    messages = build_messages(variant, question, chunks)  # validates the variant
    client = client or build_openai_client(settings, purpose="generate answers")
    text, usage = _create_answer(
        client,
        settings.chat_model,
        messages,
        PROMPT_VARIANTS[variant].temperature,
        extra_body=thinking_extra_body(settings),
    )
    if not text:
        raise EmptyAnswerError(f"Model returned no answer for variant {variant!r}")
    return RagAnswer(question=question, answer=text, variant=variant, sources=chunks, usage=usage)


def retrieve_context(
    settings: Settings,
    conn: psycopg.Connection,
    question: str,
    limit: int | None = None,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Retrieve the context chunks for a question via the served method."""
    results = retrieval.search(
        settings,
        conn,
        question,
        limit=settings.rag_context_size if limit is None else limit,
        embedder=embedder,
    )
    return [result.chunk for result in results]


def answer(
    settings: Settings,
    conn: psycopg.Connection,
    question: str,
    variant: str | None = None,
    limit: int | None = None,
    embedder: Embedder | None = None,
    client: OpenAI | None = None,
) -> RagAnswer:
    """Full RAG flow: retrieve context, then generate a grounded answer."""
    chosen = settings.rag_prompt_variant if variant is None else variant
    if chosen not in PROMPT_VARIANTS:  # fail before paying for retrieval
        raise ValueError(f"Unknown prompt variant {chosen!r}; expected {list(PROMPT_VARIANTS)}")
    chunks = retrieve_context(settings, conn, question, limit=limit, embedder=embedder)
    return generate_answer(settings, question, chunks, variant=chosen, client=client)
