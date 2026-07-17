"""Tests for transcript chunking."""

from cs336_rag.ingest.chunking import chunk_transcript
from cs336_rag.models import Chunk, TranscriptSegment, VideoTranscript


def make_transcript(texts: list[str], seconds_each: float = 4.0) -> VideoTranscript:
    segments = [
        TranscriptSegment(text=text, start=i * seconds_each, duration=seconds_each)
        for i, text in enumerate(texts)
    ]
    return VideoTranscript(
        video_id="vid42",
        title="Lecture 3: Architectures",
        position=3,
        source="youtube",
        segments=segments,
    )


def test_short_transcript_becomes_single_chunk() -> None:
    transcript = make_transcript(["hello there", "general kenobi"])

    chunks = chunk_transcript(transcript, max_chars=1000, overlap_chars=100)

    assert len(chunks) == 1
    assert chunks[0].content == "hello there general kenobi"
    assert chunks[0].chunk_index == 0
    assert chunks[0].start == 0.0
    assert chunks[0].end == 8.0


def test_chunk_carries_video_metadata() -> None:
    chunks = chunk_transcript(make_transcript(["a"]), max_chars=100, overlap_chars=10)

    chunk = chunks[0]
    assert chunk.video_id == "vid42"
    assert chunk.title == "Lecture 3: Architectures"
    assert chunk.position == 3
    assert chunk.id == "vid42:0"
    assert chunk.url == "https://www.youtube.com/watch?v=vid42&t=0s"


def test_long_transcript_splits_into_multiple_chunks() -> None:
    texts = [f"segment number {i} with some words" for i in range(100)]
    transcript = make_transcript(texts)

    chunks = chunk_transcript(transcript, max_chars=300, overlap_chars=60)

    assert len(chunks) > 5
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_every_segment_appears_in_some_chunk() -> None:
    texts = [f"unique-token-{i}" for i in range(50)]
    transcript = make_transcript(texts)

    chunks = chunk_transcript(transcript, max_chars=200, overlap_chars=40)

    joined = " ".join(chunk.content for chunk in chunks)
    for text in texts:
        assert text in joined


def test_consecutive_chunks_overlap() -> None:
    texts = [f"word{i}" for i in range(60)]
    transcript = make_transcript(texts)

    chunks = chunk_transcript(transcript, max_chars=150, overlap_chars=30)

    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:], strict=False):
        previous_words = previous.content.split()
        current_words = current.content.split()
        # the head of each chunk repeats the tail of the previous one
        assert current_words[0] in previous_words


def test_timestamps_are_monotonic_and_within_video() -> None:
    texts = [f"filler text {i}" for i in range(80)]
    transcript = make_transcript(texts)

    chunks = chunk_transcript(transcript, max_chars=250, overlap_chars=50)

    for chunk in chunks:
        assert chunk.start < chunk.end
    starts = [chunk.start for chunk in chunks]
    assert starts == sorted(starts)


def test_url_timestamp_matches_chunk_start() -> None:
    texts = [f"word{i}" for i in range(60)]
    chunks = chunk_transcript(make_transcript(texts), max_chars=150, overlap_chars=30)

    later_chunk = chunks[1]
    assert f"&t={int(later_chunk.start)}s" in later_chunk.url


def test_chunks_never_wildly_exceed_max_chars() -> None:
    texts = ["short bit of speech here"] * 100
    chunks = chunk_transcript(make_transcript(texts), max_chars=300, overlap_chars=60)

    for chunk in chunks:
        # one segment of slack is allowed past max_chars
        assert len(chunk.content) <= 300 + len("short bit of speech here") + 1


def test_chunk_is_a_model_with_expected_fields() -> None:
    chunk = Chunk(
        video_id="v",
        title="t",
        position=1,
        chunk_index=2,
        start=10.0,
        end=20.0,
        content="c",
    )
    assert chunk.id == "v:2"
