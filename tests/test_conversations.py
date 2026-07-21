"""Integration tests for conversation and feedback logging."""

import psycopg
import pytest

from cs336_rag import conversations
from cs336_rag.conversations import ConversationRecord
from tests.conftest import make_chunk

pytestmark = pytest.mark.integration


def make_record(**overrides: object) -> ConversationRecord:
    data: dict[str, object] = {
        "question": "what is bpe?",
        "answer": "Byte pair encoding merges frequent pairs [1].",
        "variant": "tutor",
        "retrieval_method": "vector",
        "source_ids": ["vid1:0", "vid1:1"],
        "retrieval_ms": 12.5,
        "generation_ms": 900.0,
        "total_ms": 912.5,
        "prompt_tokens": 1200,
        "completion_tokens": 150,
    }
    data.update(overrides)
    return ConversationRecord(**data)  # type: ignore[arg-type]


class TestLogConversation:
    def test_returns_id_and_persists(self, db_conn: psycopg.Connection) -> None:
        conversation_id = conversations.log_conversation(db_conn, make_record())

        row = db_conn.execute(
            "SELECT question, answer, variant, retrieval_method, num_sources "
            "FROM conversations WHERE id = %s",
            (conversation_id,),
        ).fetchone()
        assert row == ("what is bpe?", "Byte pair encoding merges frequent pairs [1].",
                       "tutor", "vector", 2)

    def test_stores_timings_and_tokens(self, db_conn: psycopg.Connection) -> None:
        conversation_id = conversations.log_conversation(db_conn, make_record())

        row = db_conn.execute(
            "SELECT total_ms, prompt_tokens, completion_tokens FROM conversations WHERE id = %s",
            (conversation_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(912.5)
        assert row[1] == 1200
        assert row[2] == 150

    def test_ids_are_unique(self, db_conn: psycopg.Connection) -> None:
        first = conversations.log_conversation(db_conn, make_record())
        second = conversations.log_conversation(db_conn, make_record())
        assert first != second

    def test_optional_token_counts(self, db_conn: psycopg.Connection) -> None:
        conversation_id = conversations.log_conversation(
            db_conn, make_record(prompt_tokens=None, completion_tokens=None)
        )
        row = db_conn.execute(
            "SELECT prompt_tokens FROM conversations WHERE id = %s", (conversation_id,)
        ).fetchone()
        assert row == (None,)


class TestFeedback:
    def test_records_positive_and_negative(self, db_conn: psycopg.Connection) -> None:
        conversation_id = conversations.log_conversation(db_conn, make_record())

        conversations.add_feedback(db_conn, conversation_id, 1)

        row = db_conn.execute(
            "SELECT rating FROM feedback WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        assert row == (1,)

    def test_unknown_conversation_raises(self, db_conn: psycopg.Connection) -> None:
        from uuid import uuid4

        with pytest.raises(conversations.UnknownConversationError):
            conversations.add_feedback(db_conn, uuid4(), 1)

    def test_invalid_rating_raises(self, db_conn: psycopg.Connection) -> None:
        conversation_id = conversations.log_conversation(db_conn, make_record())
        with pytest.raises(ValueError, match="rating"):
            conversations.add_feedback(db_conn, conversation_id, 5)

    def test_feedback_replaces_previous_vote(self, db_conn: psycopg.Connection) -> None:
        conversation_id = conversations.log_conversation(db_conn, make_record())

        conversations.add_feedback(db_conn, conversation_id, 1)
        conversations.add_feedback(db_conn, conversation_id, -1)

        rows = db_conn.execute(
            "SELECT rating FROM feedback WHERE conversation_id = %s", (conversation_id,)
        ).fetchall()
        assert rows == [(-1,)]


class TestStats:
    def test_empty_database(self, db_conn: psycopg.Connection) -> None:
        stats = conversations.get_stats(db_conn)
        assert stats.conversations == 0
        assert stats.positive == 0
        assert stats.negative == 0

    def test_counts_conversations_and_votes(self, db_conn: psycopg.Connection) -> None:
        first = conversations.log_conversation(db_conn, make_record())
        second = conversations.log_conversation(db_conn, make_record())
        conversations.log_conversation(db_conn, make_record())
        conversations.add_feedback(db_conn, first, 1)
        conversations.add_feedback(db_conn, second, -1)

        stats = conversations.get_stats(db_conn)

        assert stats.conversations == 3
        assert stats.positive == 1
        assert stats.negative == 1
        assert stats.avg_total_ms == pytest.approx(912.5)


def test_chunk_source_ids_roundtrip(db_conn: psycopg.Connection) -> None:
    chunks = [make_chunk(0), make_chunk(1)]
    record = make_record(source_ids=[c.id for c in chunks])

    conversation_id = conversations.log_conversation(db_conn, record)

    row = db_conn.execute(
        "SELECT source_ids FROM conversations WHERE id = %s", (conversation_id,)
    ).fetchone()
    assert row == (["vid1:0", "vid1:1"],)
