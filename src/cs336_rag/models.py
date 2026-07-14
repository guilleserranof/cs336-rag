"""Domain models shared across ingestion, retrieval and the API."""

from typing import Literal

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """One caption snippet as returned by YouTube or Whisper."""

    text: str
    start: float
    duration: float


class VideoTranscript(BaseModel):
    """Full transcript of a single lecture video."""

    video_id: str
    title: str
    position: int
    source: Literal["youtube", "whisper"]
    segments: list[TranscriptSegment]

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def full_text(self) -> str:
        return " ".join(segment.text for segment in self.segments)
