"""Tests for transcript fetching and persistence."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cs336_rag.config import Settings
from cs336_rag.ingest.transcripts import (
    fetch_all_transcripts,
    fetch_youtube_transcript,
    load_transcripts,
    save_transcript,
)
from cs336_rag.models import TranscriptSegment, VideoTranscript


def make_transcript(**overrides: Any) -> VideoTranscript:
    data: dict[str, Any] = {
        "video_id": "abc123",
        "title": "Lecture 1: Overview, Tokenization",
        "position": 1,
        "source": "youtube",
        "segments": [
            TranscriptSegment(text="hello", start=0.0, duration=2.0),
            TranscriptSegment(text="world", start=2.0, duration=3.5),
        ],
    }
    data.update(overrides)
    return VideoTranscript(**data)


class TestVideoTranscript:
    def test_url_points_to_video(self) -> None:
        assert make_transcript().url == "https://www.youtube.com/watch?v=abc123"

    def test_full_text_joins_segments(self) -> None:
        assert make_transcript().full_text == "hello world"


class TestFetchYoutubeTranscript:
    def _api_with(self, transcript: MagicMock | None, generated: MagicMock | None) -> MagicMock:
        """Build a fake YouTubeTranscriptApi whose list() returns our transcripts."""
        from youtube_transcript_api import NoTranscriptFound

        api = MagicMock()
        transcript_list = MagicMock()
        exc = NoTranscriptFound("abc123", ["en"], MagicMock())
        if transcript is None:
            transcript_list.find_manually_created_transcript.side_effect = exc
        else:
            transcript_list.find_manually_created_transcript.return_value = transcript
        if generated is None:
            transcript_list.find_generated_transcript.side_effect = exc
        else:
            transcript_list.find_generated_transcript.return_value = generated
        api.list.return_value = transcript_list
        return api

    def _fake_fetched(self) -> MagicMock:
        snippet = MagicMock()
        snippet.text = "hi there"
        snippet.start = 1.0
        snippet.duration = 2.0
        fetched = MagicMock()
        fetched.snippets = [snippet]
        return fetched

    def test_prefers_manual_transcript(self) -> None:
        manual = MagicMock()
        manual.fetch.return_value = self._fake_fetched()
        generated = MagicMock()
        api = self._api_with(manual, generated)

        segments = fetch_youtube_transcript("abc123", api=api)

        assert segments == [TranscriptSegment(text="hi there", start=1.0, duration=2.0)]
        generated.fetch.assert_not_called()

    def test_falls_back_to_generated(self) -> None:
        generated = MagicMock()
        generated.fetch.return_value = self._fake_fetched()
        api = self._api_with(None, generated)

        segments = fetch_youtube_transcript("abc123", api=api)

        assert segments is not None
        assert len(segments) == 1

    def test_returns_none_when_no_transcript(self) -> None:
        api = self._api_with(None, None)
        assert fetch_youtube_transcript("abc123", api=api) is None


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        transcript = make_transcript()
        save_transcript(transcript, tmp_path)

        loaded = load_transcripts(tmp_path)

        assert loaded == [transcript]

    def test_saved_file_is_readable_json(self, tmp_path: Path) -> None:
        save_transcript(make_transcript(), tmp_path)
        raw = json.loads((tmp_path / "001-abc123.json").read_text())
        assert raw["video_id"] == "abc123"
        assert len(raw["segments"]) == 2

    def test_load_sorted_by_position(self, tmp_path: Path) -> None:
        save_transcript(make_transcript(video_id="v2", position=2), tmp_path)
        save_transcript(make_transcript(video_id="v1", position=1), tmp_path)

        loaded = load_transcripts(tmp_path)

        assert [t.position for t in loaded] == [1, 2]

    def test_load_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_transcripts(tmp_path / "nope")


class TestFetchAllTranscripts:
    def test_existing_transcripts_do_not_need_openai_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]
        transcript = make_transcript()
        save_transcript(transcript, settings.raw_transcripts_dir)
        monkeypatch.setattr(
            "cs336_rag.ingest.transcripts.list_playlist_videos",
            lambda playlist_id: [(transcript.video_id, transcript.title)],
        )
        monkeypatch.setattr(
            "cs336_rag.ingest.transcripts.fetch_youtube_transcript",
            MagicMock(side_effect=AssertionError("should not re-fetch existing transcript")),
        )

        assert fetch_all_transcripts(settings) == [transcript]
