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

    generate = subparsers.add_parser(
        "generate-ground-truth",
        help="Generate evaluation questions from a sample of chunks (LLM) into data/.",
    )
    generate.add_argument("--sample", type=int, default=150, help="Chunks to sample.")
    generate.add_argument("--per-chunk", type=int, default=2, help="Questions per chunk.")
    generate.add_argument("--seed", type=int, default=42, help="Sampling seed.")

    evaluate = subparsers.add_parser(
        "evaluate-retrieval",
        help="Score all retrieval methods (hit rate@5/@10, MRR) on the ground truth.",
    )
    evaluate.add_argument("--limit", type=int, default=10, help="Results per query.")

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
    elif args.command == "generate-ground-truth":
        from cs336_rag.config import get_settings
        from cs336_rag.evals.ground_truth import (
            generate_ground_truth,
            sample_chunks,
            save_ground_truth,
        )
        from cs336_rag.ingest.chunking import chunk_transcripts
        from cs336_rag.ingest.transcripts import load_transcripts
        from cs336_rag.llm import build_openai_client

        settings = get_settings()
        transcripts = load_transcripts(settings.raw_transcripts_dir)
        chunks = chunk_transcripts(
            transcripts, settings.chunk_max_chars, settings.chunk_overlap_chars
        )
        sampled = sample_chunks(chunks, size=args.sample, seed=args.seed)
        client = build_openai_client(settings, purpose="generate ground-truth questions")
        entries = generate_ground_truth(
            sampled, client, model=settings.chat_model, per_chunk=args.per_chunk
        )
        path = settings.data_dir / "ground_truth.json"
        save_ground_truth(entries, path)
        print(f"Generated {len(entries)} questions from {len(sampled)} chunks -> {path}")
    elif args.command == "evaluate-retrieval":
        from cs336_rag import db
        from cs336_rag.config import get_settings
        from cs336_rag.embeddings import EmbeddingClient
        from cs336_rag.evals.ground_truth import load_ground_truth
        from cs336_rag.evals.retrieval_eval import evaluate_retrieval
        from cs336_rag.models import ALL_SEARCH_METHODS

        settings = get_settings()
        entries = load_ground_truth(settings.data_dir / "ground_truth.json")
        methods = list(ALL_SEARCH_METHODS)
        with db.connect(settings) as conn:
            report = evaluate_retrieval(
                settings,
                conn,
                entries,
                methods=methods,
                embedder=EmbeddingClient(settings),
                limit=args.limit,
            )
        path = settings.data_dir / "eval" / "retrieval_eval.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(report.as_markdown())
        print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
