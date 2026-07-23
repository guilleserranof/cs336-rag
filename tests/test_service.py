"""Tests for the RagService, including optional query rewriting."""

from unittest.mock import MagicMock

from cs336_rag import rag, service
from cs336_rag.service import RagService
from tests.conftest import make_chunk
from tests.test_config import make_settings


def make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    return embedder


class TestQueryRewriting:
    def test_rewrites_search_query_when_enabled(self, monkeypatch: object) -> None:
        import pytest

        assert isinstance(monkeypatch, pytest.MonkeyPatch)
        seen: dict[str, str] = {}

        def fake_retrieve(settings, conn, question, limit=None, embedder=None):  # type: ignore[no-untyped-def]
            seen["search_query"] = question
            return [make_chunk(0)]

        monkeypatch.setattr(service, "retrieve_context", fake_retrieve)
        monkeypatch.setattr(
            service, "rewrite_query", lambda settings, q, client=None: "rewritten: " + q
        )

        svc = RagService(
            make_settings(rag_rewrite_query=True), embedder=make_embedder(), client=MagicMock()
        )
        svc.retrieve(MagicMock(), "whats bpe")

        assert seen["search_query"] == "rewritten: whats bpe"

    def test_uses_original_query_when_disabled(self, monkeypatch: object) -> None:
        import pytest

        assert isinstance(monkeypatch, pytest.MonkeyPatch)
        seen: dict[str, str] = {}

        def fake_retrieve(settings, conn, question, limit=None, embedder=None):  # type: ignore[no-untyped-def]
            seen["search_query"] = question
            return [make_chunk(0)]

        monkeypatch.setattr(service, "retrieve_context", fake_retrieve)
        monkeypatch.setattr(
            service,
            "rewrite_query",
            MagicMock(side_effect=AssertionError("must not rewrite when disabled")),
        )

        svc = RagService(
            make_settings(rag_rewrite_query=False), embedder=make_embedder(), client=MagicMock()
        )
        svc.retrieve(MagicMock(), "whats bpe")

        assert seen["search_query"] == "whats bpe"

    def test_answer_prompt_keeps_the_original_question(self, monkeypatch: object) -> None:
        # retrieval may use a rewritten query, but the answer must address the
        # question the user actually asked.
        import pytest

        assert isinstance(monkeypatch, pytest.MonkeyPatch)
        monkeypatch.setattr(service, "retrieve_context", lambda *a, **k: [make_chunk(0)])
        monkeypatch.setattr(service, "rewrite_query", lambda settings, q, client=None: "REWRITE")
        captured: dict[str, str] = {}

        def fake_generate(settings, question, chunks, variant, client=None):  # type: ignore[no-untyped-def]
            captured["question"] = question
            return rag.RagAnswer(question=question, answer="a", variant=variant, sources=chunks)

        monkeypatch.setattr(service, "generate_answer", fake_generate)

        svc = RagService(
            make_settings(rag_rewrite_query=True), embedder=make_embedder(), client=MagicMock()
        )
        _, chunks = svc.retrieve(MagicMock(), "original question")
        svc.generate("original question", [make_chunk(0)])

        assert captured["question"] == "original question"


def test_default_rag_rewrite_query_is_off() -> None:
    assert make_settings().rag_rewrite_query is False
