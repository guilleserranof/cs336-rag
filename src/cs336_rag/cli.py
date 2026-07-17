"""Command-line entry point: ``uv run cs336-rag <command>``."""

import argparse
import logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cs336-rag",
        description="RAG assistant over the Stanford CS336 lecture series.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch-transcripts",
        help="Download lecture transcripts from YouTube (Whisper fallback) into data/raw/.",
    )
    fetch.add_argument(
        "--force", action="store_true", help="Re-fetch transcripts that already exist."
    )

    subparsers.add_parser(
        "ingest",
        help="Chunk transcripts, embed them and (re)load the Postgres knowledge base.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "fetch-transcripts":
        from cs336_rag.config import get_settings
        from cs336_rag.ingest.transcripts import fetch_all_transcripts

        transcripts = fetch_all_transcripts(get_settings(), force=args.force)
        total_segments = sum(len(transcript.segments) for transcript in transcripts)
        print(f"Fetched {len(transcripts)} transcripts ({total_segments} segments).")
    elif args.command == "ingest":
        from cs336_rag.config import get_settings
        from cs336_rag.ingest.pipeline import run_ingestion

        stats = run_ingestion(get_settings())
        print(f"Ingested {stats.videos} videos as {stats.chunks} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
