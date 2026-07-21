"""Domain models shared across ingestion, retrieval and the API."""

from typing import Literal, get_args

from pydantic import BaseModel

YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v="

# Retrieval methods compared in the evaluation (cs336_rag.evals.retrieval_eval).
SearchMethod = Literal["text", "vector", "hybrid", "hybrid_rerank"]
ALL_SEARCH_METHODS: tuple[SearchMethod, ...] = get_args(SearchMethod)

# RAG prompt variants compared in the answer evaluation (cs336_rag.evals.answer_eval).
# The templates live in cs336_rag.rag; the names live here so Settings can
# validate them without importing the generation layer.
PromptVariantName = Literal["baseline", "grounded", "tutor"]
ALL_PROMPT_VARIANTS: tuple[PromptVariantName, ...] = get_args(PromptVariantName)


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
        return f"{YOUTUBE_WATCH_URL}{self.video_id}"

    @property
    def full_text(self) -> str:
        return " ".join(segment.text for segment in self.segments)


class Chunk(BaseModel):
    """A retrievable slice of a lecture, deep-linkable via the YouTube URL schema."""

    video_id: str
    title: str
    position: int
    chunk_index: int
    start: float
    end: float
    content: str

    @property
    def id(self) -> str:
        return f"{self.video_id}:{self.chunk_index}"

    @property
    def url(self) -> str:
        """Deep link that opens the video at the chunk's first second."""
        return f"{YOUTUBE_WATCH_URL}{self.video_id}&t={int(self.start)}s"
