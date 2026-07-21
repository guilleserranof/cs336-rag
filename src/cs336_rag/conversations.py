"""Persistence for application telemetry: conversations and user feedback.

Every answered question is logged with its timings, token usage and the
sources used. Feedback is a single vote per conversation (a second vote
replaces the first). These tables are what the Grafana dashboards read.
"""

import logging
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

logger = logging.getLogger(__name__)

VALID_RATINGS = (-1, 1)


class UnknownConversationError(LookupError):
    """Feedback referenced a conversation that does not exist."""


class ConversationRecord(BaseModel):
    """One answered question, as logged for monitoring."""

    question: str
    answer: str
    variant: str
    retrieval_method: str
    source_ids: list[str] = []
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Stats(BaseModel):
    conversations: int
    positive: int
    negative: int
    avg_total_ms: float | None = None


def log_conversation(conn: psycopg.Connection, record: ConversationRecord) -> UUID:
    """Persist an answered question and return its id."""
    conversation_id = uuid4()
    conn.execute(
        """
        INSERT INTO conversations (
            id, question, answer, variant, retrieval_method, source_ids, num_sources,
            retrieval_ms, generation_ms, total_ms, prompt_tokens, completion_tokens
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            conversation_id,
            record.question,
            record.answer,
            record.variant,
            record.retrieval_method,
            record.source_ids,
            len(record.source_ids),
            record.retrieval_ms,
            record.generation_ms,
            record.total_ms,
            record.prompt_tokens,
            record.completion_tokens,
        ),
    )
    conn.commit()
    return conversation_id


def add_feedback(conn: psycopg.Connection, conversation_id: UUID, rating: int) -> None:
    """Record a vote for a conversation, replacing any previous vote."""
    if rating not in VALID_RATINGS:
        raise ValueError(f"Invalid rating {rating!r}; expected one of {VALID_RATINGS}")
    try:
        conn.execute(
            """
            INSERT INTO feedback (conversation_id, rating) VALUES (%s, %s)
            ON CONFLICT (conversation_id)
            DO UPDATE SET rating = EXCLUDED.rating, created_at = now()
            """,
            (conversation_id, rating),
        )
    except psycopg.errors.ForeignKeyViolation as error:
        conn.rollback()
        raise UnknownConversationError(f"No conversation {conversation_id}") from error
    conn.commit()


def get_stats(conn: psycopg.Connection) -> Stats:
    """Aggregate counters for the UI (Grafana queries the tables directly)."""
    with conn.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute(
            """
            SELECT
                (SELECT count(*) FROM conversations)                      AS conversations,
                (SELECT count(*) FROM feedback WHERE rating = 1)          AS positive,
                (SELECT count(*) FROM feedback WHERE rating = -1)         AS negative,
                (SELECT avg(total_ms) FROM conversations)                 AS avg_total_ms
            """
        ).fetchone()
    assert row is not None
    return Stats(
        conversations=row["conversations"],
        positive=row["positive"],
        negative=row["negative"],
        avg_total_ms=float(row["avg_total_ms"]) if row["avg_total_ms"] is not None else None,
    )
