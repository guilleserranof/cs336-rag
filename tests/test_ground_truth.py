"""Tests for LLM ground-truth question generation."""

from pathlib import Path
from unittest.mock import MagicMock

from cs336_rag.evals.ground_truth import (
    GroundTruthEntry,
    generate_ground_truth,
    load_ground_truth,
    parse_questions,
    sample_chunks,
    save_ground_truth,
)
from tests.conftest import make_chunk


class TestParseQuestions:
    def test_clean_json_array(self) -> None:
        assert parse_questions('["q1?", "q2?"]') == ["q1?", "q2?"]

    def test_fenced_json_array(self) -> None:
        raw = 'Here you go:\n```json\n["what is bpe?", "why merge pairs?"]\n```'
        assert parse_questions(raw) == ["what is bpe?", "why merge pairs?"]

    def test_garbage_returns_empty(self) -> None:
        assert parse_questions("no questions here") == []

    def test_non_string_items_are_dropped(self) -> None:
        assert parse_questions('["ok?", 42, null]') == ["ok?"]

    def test_stray_brackets_in_prose_do_not_break_parsing(self) -> None:
        raw = 'Here are questions [based on lecture 3]:\n["q1?", "q2?"]'
        assert parse_questions(raw) == ["q1?", "q2?"]

    def test_bracket_inside_question_string_is_preserved(self) -> None:
        assert parse_questions('["what is BPE [tokenization]?"]') == ["what is BPE [tokenization]?"]


class TestSampleChunks:
    def test_deterministic_with_seed(self) -> None:
        chunks = [make_chunk(i, f"content {i}") for i in range(50)]
        first = sample_chunks(chunks, size=10, seed=42)
        second = sample_chunks(chunks, size=10, seed=42)
        assert [c.id for c in first] == [c.id for c in second]

    def test_sample_size_capped_at_population(self) -> None:
        chunks = [make_chunk(i) for i in range(3)]
        assert len(sample_chunks(chunks, size=10, seed=1)) == 3


class TestGenerateGroundTruth:
    def _chat_returning(self, payloads: list[str]) -> MagicMock:
        client = MagicMock()
        completions = []
        for payload in payloads:
            completion = MagicMock()
            completion.choices = [MagicMock()]
            completion.choices[0].message.content = payload
            completions.append(completion)
        client.chat.completions.create.side_effect = completions
        return client

    def test_generates_entries_per_chunk(self) -> None:
        chunks = [make_chunk(0, "tokenizers"), make_chunk(1, "attention")]
        client = self._chat_returning(['["q1?", "q2?"]', '["q3?", "q4?"]'])

        entries = generate_ground_truth(chunks, client, model="qwen3.6", per_chunk=2)

        assert len(entries) == 4
        assert entries[0] == GroundTruthEntry(question="q1?", chunk_id="vid1:0")
        assert entries[2].chunk_id == "vid1:1"

    def test_unparseable_chunk_is_skipped(self) -> None:
        chunks = [make_chunk(0), make_chunk(1)]
        client = self._chat_returning(["not json at all", '["good?"]'])

        entries = generate_ground_truth(chunks, client, model="qwen3.6", per_chunk=1)

        assert [entry.chunk_id for entry in entries] == ["vid1:1"]


class TestPersistence:
    def test_roundtrip(self, tmp_path: Path) -> None:
        entries = [
            GroundTruthEntry(question="what is bpe?", chunk_id="vid1:0"),
            GroundTruthEntry(question="why attention?", chunk_id="vid1:1"),
        ]
        path = tmp_path / "ground_truth.json"

        save_ground_truth(entries, path)
        loaded = load_ground_truth(path)

        assert loaded == entries
