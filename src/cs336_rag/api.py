"""FastAPI application: ask questions, vote on answers, serve the web UI.

Every answered question is logged to Postgres with its timings and token
usage, and feedback is recorded against it — that history is what the
Grafana dashboards chart.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from typing import Annotated, Literal, Protocol
from uuid import UUID

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from cs336_rag import conversations, db
from cs336_rag.config import Settings, get_settings
from cs336_rag.conversations import ConversationRecord, Stats, UnknownConversationError
from cs336_rag.models import Chunk
from cs336_rag.service import AnsweredResult, RagService

logger = logging.getLogger(__name__)


class Answerer(Protocol):
    """What the API needs from the RAG service (real or fake)."""

    def answer(
        self, conn: psycopg.Connection, question: str, variant: str | None = ...
    ) -> AnsweredResult: ...


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


def create_app(settings: Settings | None = None, service: Answerer | None = None) -> FastAPI:
    """Build the application.

    ``service`` is injectable so tests can exercise the endpoints without
    calling the LLM; in production it is constructed once at startup.
    """
    settings = settings or get_settings()
    app = FastAPI(
        title="CS336 Lecture Assistant",
        description="Ask questions about the Stanford CS336 lecture series.",
        version="0.1.0",
    )

    @contextmanager
    def connection() -> Iterator[psycopg.Connection]:
        conn = db.connect(settings)
        try:
            yield conn
        finally:
            conn.close()

    def get_service() -> Answerer:
        if app.state.service is None:  # built lazily so import never needs a key
            app.state.service = RagService(settings)
        return app.state.service  # type: ignore[no-any-return]

    app.state.service = service

    # The telemetry tables must exist before the first request. Ingestion may
    # never have run on this database, and it is the app that owns these tables.
    with connection() as conn:
        db.init_app_schema(conn)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ask", response_model=AskResponse)
    def ask(
        request: AskRequest, answerer: Annotated[Answerer, Depends(get_service)]
    ) -> AskResponse:
        with connection() as conn:
            result = answerer.answer(conn, request.question, request.variant)
            answer = result.answer
            usage = answer.usage
            conversation_id = conversations.log_conversation(
                conn,
                ConversationRecord(
                    question=answer.question,
                    answer=answer.answer,
                    variant=answer.variant,
                    retrieval_method=settings.retrieval_method,
                    source_ids=[chunk.id for chunk in answer.sources],
                    retrieval_ms=result.retrieval_ms,
                    generation_ms=result.generation_ms,
                    total_ms=result.total_ms,
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
            retrieval_ms=result.retrieval_ms,
            generation_ms=result.generation_ms,
            total_ms=result.total_ms,
        )

    @app.post("/api/feedback")
    def feedback(request: FeedbackRequest) -> dict[str, str]:
        with connection() as conn:
            try:
                conversations.add_feedback(conn, request.conversation_id, request.rating)
            except UnknownConversationError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
        return {"status": "recorded"}

    @app.get("/api/stats", response_model=Stats)
    def stats() -> Stats:
        with connection() as conn:
            return conversations.get_stats(conn)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = resources.files("cs336_rag").joinpath("static/index.html").read_text("utf-8")
        return HTMLResponse(html)

    return app


app = create_app  # uvicorn factory target: `uvicorn cs336_rag.api:app --factory`
