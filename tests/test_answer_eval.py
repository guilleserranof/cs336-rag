"""Tests for LLM-judged prompt evaluation."""

from unittest.mock import MagicMock

import pytest

from cs336_rag.evals.answer_eval import (
    JudgeScore,
    evaluate_prompts,
    judge_answer,
    parse_judge,
)
from cs336_rag.models import Chunk
from tests.conftest import make_chunk
from tests.test_config import make_settings


class TestParseJudge:
    def test_parses_clean_json(self) -> None:
        score = parse_judge('{"relevance": 5, "groundedness": 4}')
        assert score == JudgeScore(relevance=5, groundedness=4)

    def test_parses_fenced_json_with_prose(self) -> None:
        raw = 'My assessment:\n```json\n{"relevance": 3, "groundedness": 2}\n```'
        assert parse_judge(raw) == JudgeScore(relevance=3, groundedness=2)

    def test_clamps_out_of_range_scores(self) -> None:
        score = parse_judge('{"relevance": 9, "groundedness": 0}')
        assert score == JudgeScore(relevance=5, groundedness=1)

    def test_garbage_returns_none(self) -> None:
        assert parse_judge("no json here") is None

    def test_missing_field_returns_none(self) -> None:
        assert parse_judge('{"relevance": 4}') is None


class TestJudgeScore:
    def test_overall_is_mean(self) -> None:
        assert JudgeScore(relevance=5, groundedness=3).overall == 4.0


class TestJudgeAnswer:
    def _judge_client(self, payload: str) -> MagicMock:
        client = MagicMock()
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = payload
        client.chat.completions.create.return_value = completion
        return client

    def test_returns_parsed_score(self) -> None:
        client = self._judge_client('{"relevance": 4, "groundedness": 5}')
        score = judge_answer(
            make_settings(), "q", [make_chunk(0, "ctx")], "an answer", client=client
        )
        assert score == JudgeScore(relevance=4, groundedness=5)

    def test_uses_judge_model(self) -> None:
        client = self._judge_client('{"relevance": 4, "groundedness": 5}')
        judge_answer(
            make_settings(judge_model="deepseek-v4-flash"),
            "q",
            [make_chunk(0)],
            "a",
            client=client,
        )
        assert client.chat.completions.create.call_args.kwargs["model"] == "deepseek-v4-flash"

    def test_unparseable_judgement_returns_none(self) -> None:
        client = self._judge_client("I cannot rate this")
        assert judge_answer(make_settings(), "q", [make_chunk(0)], "a", client=client) is None


class TestEvaluatePrompts:
    def _clients(self, judge_payload: str) -> tuple[MagicMock, MagicMock]:
        gen = MagicMock()
        gen_completion = MagicMock()
        gen_completion.choices = [MagicMock()]
        gen_completion.choices[0].message.content = "generated answer"
        gen.chat.completions.create.return_value = gen_completion

        judge = MagicMock()
        judge_completion = MagicMock()
        judge_completion.choices = [MagicMock()]
        judge_completion.choices[0].message.content = judge_payload
        judge.chat.completions.create.return_value = judge_completion
        return gen, judge

    def test_scores_each_variant_over_questions(self) -> None:
        gen, judge = self._clients('{"relevance": 5, "groundedness": 5}')
        chunks = [make_chunk(0, "ctx")]

        report = evaluate_prompts(
            make_settings(),
            questions=["q1", "q2"],
            variants=["grounded", "concise"],
            retrieve=lambda question: chunks,
            gen_client=gen,
            judge_client=judge,
        )

        assert set(report.results) == {"grounded", "concise"}
        assert report.results["grounded"].questions == 2
        assert report.results["grounded"].avg_overall == 5.0

    def _completion(self, text: str) -> MagicMock:
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = text
        return completion

    def test_best_variant_is_highest_scoring(self) -> None:
        # one question, two variants -> generate grounded then concise,
        # judge grounded high then concise low (evaluate processes variant by variant)
        gen = MagicMock()
        gen.chat.completions.create.side_effect = [
            self._completion("grounded answer"),
            self._completion("concise answer"),
        ]
        judge = MagicMock()
        judge.chat.completions.create.side_effect = [
            self._completion('{"relevance": 5, "groundedness": 5}'),
            self._completion('{"relevance": 2, "groundedness": 2}'),
        ]

        report = evaluate_prompts(
            make_settings(),
            questions=["q1"],
            variants=["grounded", "concise"],
            retrieve=lambda question: [make_chunk(0)],
            gen_client=gen,
            judge_client=judge,
        )

        assert report.best_variant == "grounded"
        assert report.results["grounded"].avg_overall == 5.0
        assert report.results["concise"].avg_overall == 2.0

    def test_skips_unjudgeable_answers(self) -> None:
        gen, judge = self._clients("not a score")
        report = evaluate_prompts(
            make_settings(),
            questions=["q1"],
            variants=["grounded"],
            retrieve=lambda question: [make_chunk(0)],
            gen_client=gen,
            judge_client=judge,
        )
        # no valid judgements -> variant scored over zero questions
        assert report.results["grounded"].questions == 0
