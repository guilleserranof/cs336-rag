"""Tests for disabling the chat model's reasoning ("thinking") mode.

qwen3.6 is a reasoning model: left enabled it spends ~95% of wall-clock
emitting internal reasoning before the first word of the answer. The
served path disables it via a vLLM chat-template kwarg.
"""

from unittest.mock import MagicMock

from cs336_rag import rag
from cs336_rag.llm import thinking_extra_body
from tests.conftest import make_chunk
from tests.test_config import make_settings


def make_chat_client(text: str = "answer") -> MagicMock:
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = text
    client.chat.completions.create.return_value = completion
    return client


class TestThinkingExtraBody:
    def test_disabled_by_default(self) -> None:
        assert make_settings().chat_disable_thinking is True

    def test_returns_template_kwarg_when_disabling(self) -> None:
        body = thinking_extra_body(make_settings(chat_disable_thinking=True))
        assert body == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_returns_none_when_thinking_allowed(self) -> None:
        assert thinking_extra_body(make_settings(chat_disable_thinking=False)) is None


class TestGenerateAnswerPassesExtraBody:
    def test_sends_extra_body_when_disabled(self) -> None:
        client = make_chat_client()
        rag.generate_answer(
            make_settings(chat_disable_thinking=True),
            "q",
            [make_chunk(0)],
            variant="tutor",
            client=client,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_omits_extra_body_when_allowed(self) -> None:
        client = make_chat_client()
        rag.generate_answer(
            make_settings(chat_disable_thinking=False),
            "q",
            [make_chunk(0)],
            variant="tutor",
            client=client,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
