"""Integration tests for the FastAPI application.

The database is real (conversations and feedback are the point of these
endpoints); the LLM is a fake service so no API calls are made.
"""

import psycopg
import pytest
from fastapi.testclient import TestClient

from cs336_rag.api import create_app
from cs336_rag.config import Settings
from cs336_rag.models import Chunk
from cs336_rag.rag import EmptyAnswerError, RagAnswer
from tests.conftest import make_chunk

pytestmark = pytest.mark.integration


class FakeService:
    """Stands in for RagService; records calls and returns a canned answer."""

    def __init__(self, answer_text: str = "BPE merges frequent pairs [1].") -> None:
        self.answer_text = answer_text
        self.calls: list[tuple[str, str | None]] = []

    def retrieve(self, conn: psycopg.Connection, question: str) -> tuple[list[Chunk], float]:
        return [make_chunk(0), make_chunk(1)], 10.0

    def generate(
        self, question: str, chunks: list[Chunk], variant: str | None = None
    ) -> tuple[RagAnswer, float]:
        self.calls.append((question, variant))
        if not self.answer_text:
            raise EmptyAnswerError("no answer")
        return (
            RagAnswer(
                question=question,
                answer=self.answer_text,
                variant=variant or "tutor",
                sources=chunks,
            ),
            500.0,
        )


@pytest.fixture
def client(db_settings: Settings, db_conn: psycopg.Connection) -> TestClient:
    service = FakeService()
    app = create_app(db_settings, service=service)
    test_client = TestClient(app)
    test_client.service = service  # type: ignore[attr-defined]
    return test_client


class TestHealth:
    def test_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestAsk:
    def test_returns_answer_and_sources(self, client: TestClient) -> None:
        response = client.post("/api/ask", json={"question": "what is bpe?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "BPE merges frequent pairs [1]."
        assert body["variant"] == "tutor"
        assert len(body["sources"]) == 2
        assert body["sources"][0]["url"].startswith("https://www.youtube.com/watch")
        assert body["conversation_id"]

    def test_logs_the_conversation(self, client: TestClient, db_conn: psycopg.Connection) -> None:
        response = client.post("/api/ask", json={"question": "what is bpe?"})
        conversation_id = response.json()["conversation_id"]

        row = db_conn.execute(
            "SELECT question, num_sources FROM conversations WHERE id = %s", (conversation_id,)
        ).fetchone()
        assert row == ("what is bpe?", 2)

    def test_passes_variant_through(self, client: TestClient) -> None:
        client.post("/api/ask", json={"question": "q", "variant": "grounded"})
        assert client.service.calls[-1] == ("q", "grounded")  # type: ignore[attr-defined]

    def test_blank_question_rejected(self, client: TestClient) -> None:
        assert client.post("/api/ask", json={"question": "   "}).status_code == 422

    def test_missing_question_rejected(self, client: TestClient) -> None:
        assert client.post("/api/ask", json={}).status_code == 422

    def test_unknown_variant_rejected_with_422(self, client: TestClient) -> None:
        response = client.post("/api/ask", json={"question": "q", "variant": "bogus"})
        assert response.status_code == 422

    def test_empty_generation_returns_502(
        self, db_settings: Settings, db_conn: psycopg.Connection
    ) -> None:
        app = create_app(db_settings, service=FakeService(answer_text=""))
        with TestClient(app) as failing_client:
            response = failing_client.post("/api/ask", json={"question": "q"})
        assert response.status_code == 502


class TestFeedback:
    def _ask(self, client: TestClient) -> str:
        return client.post("/api/ask", json={"question": "q"}).json()["conversation_id"]

    def test_accepts_positive_vote(self, client: TestClient, db_conn: psycopg.Connection) -> None:
        conversation_id = self._ask(client)

        response = client.post(
            "/api/feedback", json={"conversation_id": conversation_id, "rating": 1}
        )

        assert response.status_code == 200
        row = db_conn.execute(
            "SELECT rating FROM feedback WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        assert row == (1,)

    def test_unknown_conversation_returns_404(self, client: TestClient) -> None:
        from uuid import uuid4

        response = client.post("/api/feedback", json={"conversation_id": str(uuid4()), "rating": 1})
        assert response.status_code == 404

    def test_invalid_rating_returns_422(self, client: TestClient) -> None:
        conversation_id = self._ask(client)
        response = client.post(
            "/api/feedback", json={"conversation_id": conversation_id, "rating": 3}
        )
        assert response.status_code == 422


class TestStats:
    def test_reports_counts(self, client: TestClient) -> None:
        conversation_id = client.post("/api/ask", json={"question": "q"}).json()["conversation_id"]
        client.post("/api/feedback", json={"conversation_id": conversation_id, "rating": 1})

        body = client.get("/api/stats").json()

        assert body["conversations"] == 1
        assert body["positive"] == 1


class TestSchemaBootstrap:
    def test_creates_telemetry_tables_when_missing(
        self, db_settings: Settings, db_conn: psycopg.Connection
    ) -> None:
        """The app must work on a database where ingestion never ran."""
        db_conn.execute("DROP TABLE IF EXISTS feedback, conversations CASCADE")
        db_conn.commit()

        app = create_app(db_settings, service=FakeService())
        with TestClient(app) as fresh_client:
            response = fresh_client.post("/api/ask", json={"question": "q"})

        assert response.status_code == 200

    def test_does_not_touch_the_knowledge_base(
        self, db_settings: Settings, db_conn: psycopg.Connection
    ) -> None:
        """Starting the app must never drop or clear `chunks`."""
        from cs336_rag import db as db_module

        db_module.replace_chunks(db_conn, [make_chunk(0)], [[0.1] * db_settings.embedding_dim])

        create_app(db_settings, service=FakeService())

        assert db_module.count_chunks(db_conn) == 1


class TestUi:
    def test_serves_index_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "CS336" in response.text
