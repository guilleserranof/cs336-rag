"""Split transcripts into overlapping, timestamped chunks.

Caption snippets are too small to retrieve on their own (a few words each),
so consecutive segments are accumulated until a chunk reaches
``max_chars``. The next chunk re-starts from the trailing segments of the
previous one (up to ``overlap_chars``) so that sentences cut at a boundary
remain findable in at least one chunk.

Chunks never span videos, and segment boundaries are respected so every
chunk maps to an exact ``start``/``end`` timestamp for deep links.
"""

from cs336_rag.models import Chunk, TranscriptSegment, VideoTranscript


def _segment_span(segments: list[TranscriptSegment]) -> tuple[float, float]:
    last = segments[-1]
    return segments[0].start, last.start + last.duration


def chunk_transcript(
    transcript: VideoTranscript, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Chunk one transcript into overlapping windows of whole segments."""
    segments = transcript.segments
    if not segments:
        return []

    chunks: list[Chunk] = []
    total = len(segments)
    window_start = 0
    while window_start < total:
        # grow the window until it reaches max_chars (or the video ends)
        window_end = window_start
        size = 0
        while window_end < total and size < max_chars:
            size += len(segments[window_end].text) + 1
            window_end += 1

        window = segments[window_start:window_end]
        start, end = _segment_span(window)
        chunks.append(
            Chunk(
                video_id=transcript.video_id,
                title=transcript.title,
                position=transcript.position,
                chunk_index=len(chunks),
                start=start,
                end=end,
                content=" ".join(segment.text for segment in window),
            )
        )
        if window_end >= total:
            break

        # walk back from the window end to build the overlap, always keeping
        # at least one segment of forward progress
        next_start = window_end
        overlap = 0
        while (
            next_start > window_start + 1
            and overlap + len(segments[next_start - 1].text) + 1 <= overlap_chars
        ):
            next_start -= 1
            overlap += len(segments[next_start].text) + 1
        window_start = next_start

    return chunks


def chunk_transcripts(
    transcripts: list[VideoTranscript], max_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Chunk every transcript, preserving playlist order."""
    return [
        chunk
        for transcript in transcripts
        for chunk in chunk_transcript(transcript, max_chars, overlap_chars)
    ]
