"""FastAPI application: ask questions, vote on answers, serve the web UI.

Every answered question is logged to Postgres with its timings and token
usage, and feedback is recorded against it — that history is what the
Grafana dashboards chart.
"""

import logging
from functools import lru_cache
from importlib import resources
from typing import Annotated, Literal, Protocol
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_delay, wait_fixed

from cs336_rag import conversations, db
from cs336_rag.config import Settings, get_settings
from cs336_rag.conversations import ConversationRecord, Stats, UnknownConversationError
from cs336_rag.models import Chunk
from cs336_rag.rag import PROMPT_VARIANTS, EmptyAnswerError, RagAnswer
from cs336_rag.service import RagService

logger = logging.getLogger(__name__)


class Answerer(Protocol):
    """What the API needs from the RAG service (real or fake)."""

    def retrieve(self, conn: psycopg.Connection, question: str) -> tuple[list[Chunk], float]: ...

    def generate(
        self, question: str, chunks: list[Chunk], variant: str | None = ...
    ) -> tuple[RagAnswer, float]: ...


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    variant: str | None = None

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    @field_validator("variant")
    @classmethod
    def known_variant(cls, value: str | None) -> str | None:
        if value is not None and value not in PROMPT_VARIANTS:
            raise ValueError(f"unknown variant; expected one of {list(PROMPT_VARIANTS)}")
        return value


class Source(BaseModel):
    id: str
    title: str
    url: str
    start: float

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "Source":
        return cls(id=chunk.id, title=chunk.title, url=chunk.url, start=chunk.start)


class AskResponse(BaseModel):
    conversation_id: UUID
    question: str
    answer: str
    variant: str
    sources: list[Source]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class FeedbackRequest(BaseModel):
    conversation_id: UUID
    rating: Literal[-1, 1]


@lru_cache(maxsize=1)
def _index_html() -> str:
    return resources.files("cs336_rag").joinpath("static/index.html").read_text("utf-8")


@retry(stop=stop_after_delay(30), wait=wait_fixed(1), reraise=True)
def _init_schema_with_retry(pool: ConnectionPool) -> None:
    """Create the telemetry tables, tolerating a database that is still
    starting up (common in docker-compose)."""
    with pool.connection() as conn:
        db.init_app_schema(conn)


def create_app(
    settings: Settings | None = None,
    service: Answerer | None = None,
    pool: ConnectionPool | None = None,
) -> FastAPI:
    """Build the application.

    ``service`` and ``pool`` are injectable so tests can exercise the
    endpoints without calling the LLM; in production both are built once
    here and shared by every request.
    """
    settings = settings or get_settings()
    app = FastAPI(
        title="CS336 Lecture Assistant",
        description="Ask questions about the Stanford CS336 lecture series.",
        version="0.1.0",
    )
    app.state.pool = pool or db.create_pool(settings)
    app.state.service = service

    # The telemetry tables must exist before the first request: ingestion may
    # never have run on this database, and the app owns these tables.
    _init_schema_with_retry(app.state.pool)

    def get_service() -> Answerer:
        if app.state.service is None:  # built lazily so import never needs a key
            app.state.service = RagService(settings)
        return app.state.service  # type: ignore[no-any-return]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ask", response_model=AskResponse)
    def ask(
        request: AskRequest, answerer: Annotated[Answerer, Depends(get_service)]
    ) -> AskResponse:
        pool: ConnectionPool = app.state.pool

        with pool.connection() as conn:
            chunks, retrieval_ms = answerer.retrieve(conn, request.question)

        # Generation can take a minute; hold no database connection across it.
        try:
            answer, generation_ms = answerer.generate(request.question, chunks, request.variant)
        except EmptyAnswerError as error:
            logger.warning("Empty generation for %r", request.question)
            raise HTTPException(status_code=502, detail="The model returned no answer.") from error

        total_ms = retrieval_ms + generation_ms
        usage = answer.usage
        with pool.connection() as conn:
            conversation_id = conversations.log_conversation(
                conn,
                ConversationRecord(
                    question=answer.question,
                    answer=answer.answer,
                    variant=answer.variant,
                    retrieval_method=settings.retrieval_method,
                    source_ids=[chunk.id for chunk in answer.sources],
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    total_ms=total_ms,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                ),
            )
        return AskResponse(
            conversation_id=conversation_id,
            question=answer.question,
            answer=answer.answer,
            variant=answer.variant,
            sources=[Source.from_chunk(chunk) for chunk in answer.sources],
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

    @app.post("/api/feedback")
    def feedback(request: FeedbackRequest) -> dict[str, str]:
        pool: ConnectionPool = app.state.pool
        with pool.connection() as conn:
            try:
                conversations.add_feedback(conn, request.conversation_id, request.rating)
            except UnknownConversationError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
        return {"status": "recorded"}

    @app.get("/api/stats", response_model=Stats)
    def stats() -> Stats:
        pool: ConnectionPool = app.state.pool
        with pool.connection() as conn:
            return conversations.get_stats(conn)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_index_html())

    return app


app = create_app  # uvicorn factory target: `uvicorn cs336_rag.api:app --factory`
