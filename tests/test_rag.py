"""Tests for the RAG answer flow."""

from unittest.mock import MagicMock

import pytest

from cs336_rag import rag
from cs336_rag.models import Chunk
from tests.conftest import make_chunk
from tests.test_config import make_settings


def make_chat_client(answer_text: str) -> MagicMock:
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = answer_text
    client.chat.completions.create.return_value = completion
    return client


class TestPromptVariants:
    def test_at_least_three_variants(self) -> None:
        assert len(rag.PROMPT_VARIANTS) >= 3

    def test_default_variant_is_registered(self) -> None:
        settings = make_settings()
        assert settings.rag_prompt_variant in rag.PROMPT_VARIANTS


class TestFormatContext:
    def test_numbers_sources_from_one(self) -> None:
        chunks = [make_chunk(0, "byte pair encoding"), make_chunk(1, "attention")]
        context = rag.format_context(chunks)
        assert "[1]" in context
        assert "[2]" in context

    def test_includes_title_and_content(self) -> None:
        context = rag.format_context([make_chunk(0, "rotary embeddings rotate q and k")])
        assert "Lecture 1" in context
        assert "rotary embeddings rotate q and k" in context


class TestBuildMessages:
    def test_contains_question_and_context(self) -> None:
        messages = rag.build_messages("grounded", "what is bpe?", [make_chunk(0, "bpe merges")])
        joined = " ".join(m["content"] for m in messages)
        assert "what is bpe?" in joined
        assert "bpe merges" in joined

    def test_variants_differ_in_system_prompt(self) -> None:
        names = list(rag.PROMPT_VARIANTS)
        system_a = rag.build_messages(names[0], "q", [make_chunk(0)])[0]["content"]
        system_b = rag.build_messages(names[1], "q", [make_chunk(0)])[0]["content"]
        assert system_a != system_b

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="variant"):
            rag.build_messages("nope", "q", [make_chunk(0)])


class TestGenerateAnswer:
    def test_returns_answer_and_sources(self) -> None:
        chunks = [make_chunk(0, "bpe merges pairs"), make_chunk(1, "attention weighs")]
        client = make_chat_client("BPE merges frequent pairs [1].")

        result = rag.generate_answer(
            make_settings(), "what is bpe?", chunks, variant="grounded", client=client
        )

        assert result.answer == "BPE merges frequent pairs [1]."
        assert result.question == "what is bpe?"
        assert result.variant == "grounded"
        assert [c.id for c in result.sources] == ["vid1:0", "vid1:1"]

    def test_uses_configured_chat_model(self) -> None:
        client = make_chat_client("ok")
        rag.generate_answer(
            make_settings(chat_model="gemma4"), "q", [make_chunk(0)], variant="grounded",
            client=client,
        )
        assert client.chat.completions.create.call_args.kwargs["model"] == "gemma4"

    def test_empty_context_still_answers(self) -> None:
        client = make_chat_client("I don't have enough information.")
        result = rag.generate_answer(
            make_settings(), "q", [], variant="grounded", client=client
        )
        assert result.sources == []
        assert result.answer


class TestAnswer:
    def test_retrieves_then_generates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chunks = [make_chunk(0, "bpe merges pairs")]
        monkeypatch.setattr(rag, "retrieve_context", lambda *a, **k: chunks)
        client = make_chat_client("BPE merges pairs [1].")

        result = rag.answer(
            make_settings(), MagicMock(), "what is bpe?", client=client, embedder=MagicMock()
        )

        assert result.answer == "BPE merges pairs [1]."
        assert [c.id for c in result.sources] == ["vid1:0"]


def test_rag_answer_is_serializable() -> None:
    ans = rag.RagAnswer(
        question="q", answer="a", variant="grounded", sources=[make_chunk(0)]
    )
    dumped = ans.model_dump()
    assert dumped["sources"][0]["video_id"] == "vid1"
    assert isinstance(ans.sources[0], Chunk)
