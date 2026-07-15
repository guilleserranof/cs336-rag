"""Fetch lecture transcripts.

Primary source: YouTube captions via ``youtube-transcript-api`` (the CS336
lectures ship manually-reviewed ``en-US`` captions). If a video has no
captions at all, the audio track is downloaded with ``yt-dlp`` and
transcribed with the Whisper endpoint of the OpenAI-compatible API.

Each transcript is persisted to ``data/raw/NNN-<video_id>.json`` so the
dataset ships with the repository and downstream steps never need YouTube
access.
"""

import logging
import tempfile
from pathlib import Path
from typing import Literal, Protocol

from openai import OpenAI
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from cs336_rag.config import Settings
from cs336_rag.models import TranscriptSegment, VideoTranscript

logger = logging.getLogger(__name__)

PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]


class PlaylistEntry(Protocol):
    """Minimal shape of a playlist item (id + title)."""

    video_id: str
    title: str


def list_playlist_videos(playlist_id: str) -> list[tuple[str, str]]:
    """Return ``(video_id, title)`` for every video in a playlist, in order."""
    import yt_dlp

    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    options = {"extract_flat": True, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or []
    return [(entry["id"], entry["title"]) for entry in entries]


def fetch_youtube_transcript(
    video_id: str, api: YouTubeTranscriptApi | None = None
) -> list[TranscriptSegment] | None:
    """Fetch captions for a video, preferring manual captions over auto-generated.

    Returns ``None`` when the video has no captions in a preferred language,
    signalling the caller to fall back to Whisper.
    """
    api = api or YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
    except TranscriptsDisabled:
        return None

    try:
        transcript = transcript_list.find_manually_created_transcript(PREFERRED_LANGUAGES)
    except NoTranscriptFound:
        try:
            transcript = transcript_list.find_generated_transcript(PREFERRED_LANGUAGES)
        except NoTranscriptFound:
            return None

    fetched = transcript.fetch()
    return [
        TranscriptSegment(text=snippet.text, start=snippet.start, duration=snippet.duration)
        for snippet in fetched.snippets
    ]


def transcribe_with_whisper(video_id: str, settings: Settings) -> list[TranscriptSegment]:
    """Fallback: download the audio track and transcribe it with Whisper."""
    import yt_dlp

    if settings.openai_key is None:
        raise ValueError("OPENAI_KEY is required to transcribe videos with Whisper")

    client = OpenAI(api_key=settings.openai_key, base_url=settings.llm_base_url)
    with tempfile.TemporaryDirectory() as tmp_dir:
        options = {
            "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": f"{tmp_dir}/{video_id}.%(ext)s",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        audio_path = next(Path(tmp_dir).glob(f"{video_id}.*"))
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=settings.whisper_model,
                file=audio_file,
                response_format="verbose_json",
            )
    segments = getattr(response, "segments", None) or []
    return [
        TranscriptSegment(
            text=segment.text.strip(),
            start=segment.start,
            duration=segment.end - segment.start,
        )
        for segment in segments
    ]


def save_transcript(transcript: VideoTranscript, directory: Path) -> Path:
    """Write one transcript to ``NNN-<video_id>.json`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{transcript.position:03d}-{transcript.video_id}.json"
    path.write_text(transcript.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_transcripts(directory: Path) -> list[VideoTranscript]:
    """Load every saved transcript, ordered by playlist position."""
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No transcript files found in {directory}")
    transcripts = [
        VideoTranscript.model_validate_json(path.read_text(encoding="utf-8")) for path in paths
    ]
    return sorted(transcripts, key=lambda transcript: transcript.position)


def fetch_all_transcripts(settings: Settings, force: bool = False) -> list[VideoTranscript]:
    """Fetch and persist transcripts for every video in the course playlist.

    Existing files are kept unless ``force`` is set, making the command
    idempotent and cheap to re-run.
    """
    directory = settings.raw_transcripts_dir
    videos = list_playlist_videos(settings.playlist_id)
    logger.info("Playlist %s has %d videos", settings.playlist_id, len(videos))

    transcripts: list[VideoTranscript] = []
    for position, (video_id, title) in enumerate(videos, start=1):
        path = directory / f"{position:03d}-{video_id}.json"
        if path.exists() and not force:
            logger.info("[%02d/%02d] %s already fetched, skipping", position, len(videos), title)
            transcripts.append(
                VideoTranscript.model_validate_json(path.read_text(encoding="utf-8"))
            )
            continue

        segments = fetch_youtube_transcript(video_id)
        source: Literal["youtube", "whisper"] = "youtube"
        if segments is None:
            logger.warning("No captions for %s, falling back to Whisper", video_id)
            segments = transcribe_with_whisper(video_id, settings)
            source = "whisper"

        transcript = VideoTranscript(
            video_id=video_id,
            title=title,
            position=position,
            source=source,
            segments=segments,
        )
        save_transcript(transcript, directory)
        logger.info(
            "[%02d/%02d] fetched %s (%d segments, %s)",
            position,
            len(videos),
            title,
            len(segments),
            source,
        )
        transcripts.append(transcript)
    return transcripts
