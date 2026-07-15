"""Tests for the embedding client wrapper."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from cs336_rag.embeddings import EmbeddingClient
from tests.test_config import make_settings


def make_openai_stub(dim: int = 4) -> MagicMock:
    """Fake OpenAI client that returns index-tagged embeddings."""

    def create(model: str, input: list[str], dimensions: int, **kwargs: Any) -> MagicMock:  # noqa: A002
        response = MagicMock()
        data = []
        for i, _text in enumerate(input):
            item = MagicMock()
            item.index = i
            item.embedding = [float(i)] * dim
            data.append(item)
        # return out of order to prove we sort by index
        response.data = list(reversed(data))
        return response

    stub = MagicMock()
    stub.embeddings.create.side_effect = create
    return stub


def test_embed_preserves_input_order() -> None:
    stub = make_openai_stub()
    client = EmbeddingClient(make_settings(), client=stub)

    vectors = client.embed(["a", "b", "c"])

    assert vectors == [[0.0] * 4, [1.0] * 4, [2.0] * 4]


def test_embed_batches_requests() -> None:
    stub = make_openai_stub()
    client = EmbeddingClient(make_settings(embed_batch_size=2), client=stub)

    client.embed(["a", "b", "c", "d", "e"])

    batch_sizes = [len(call.kwargs["input"]) for call in stub.embeddings.create.call_args_list]
    assert batch_sizes == [2, 2, 1]


def test_embed_passes_model_and_dimensions() -> None:
    stub = make_openai_stub()
    settings = make_settings(embedding_model="qwen3-embedding", embedding_dim=1024)
    client = EmbeddingClient(settings, client=stub)

    client.embed(["a"])

    kwargs = stub.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "qwen3-embedding"
    assert kwargs["dimensions"] == 1024


def test_embed_empty_list_returns_empty() -> None:
    stub = make_openai_stub()
    client = EmbeddingClient(make_settings(), client=stub)

    assert client.embed([]) == []
    stub.embeddings.create.assert_not_called()


def test_missing_key_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="OPENAI_KEY"):
        EmbeddingClient(make_settings(openai_key=None))
