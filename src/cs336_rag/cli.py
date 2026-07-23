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
    evaluate.add_argument(
        "--rewrite",
        action="store_true",
        help="Rewrite each question into a search query first (measures query rewriting).",
    )

    ask = subparsers.add_parser("ask", help="Answer a question with the RAG flow.")
    ask.add_argument("question", help="The question to answer.")
    ask.add_argument("--variant", default=None, help="Prompt variant (default: configured).")

    evaluate_prompts_parser = subparsers.add_parser(
        "evaluate-prompts",
        help="Judge RAG prompt variants over a question sample (LLM judge).",
    )
    evaluate_prompts_parser.add_argument(
        "--sample", type=int, default=40, help="Questions to evaluate."
    )
    evaluate_prompts_parser.add_argument("--seed", type=int, default=7, help="Sampling seed.")

    serve = subparsers.add_parser("serve", help="Run the web application (FastAPI + UI).")
    serve.add_argument("--host", default="0.0.0.0", help="Bind address.")  # noqa: S104
    serve.add_argument("--port", type=int, default=8000, help="Bind port.")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")

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
        from cs336_rag.llm import build_openai_client
        from cs336_rag.models import ALL_SEARCH_METHODS

        settings = get_settings()
        entries = load_ground_truth(settings.data_dir / "ground_truth.json")
        methods = list(ALL_SEARCH_METHODS)
        rewrite_client = (
            build_openai_client(settings, purpose="rewrite queries") if args.rewrite else None
        )
        with db.connect(settings) as conn:
            report = evaluate_retrieval(
                settings,
                conn,
                entries,
                methods=methods,
                embedder=EmbeddingClient(settings),
                limit=args.limit,
                rewrite_client=rewrite_client,
            )
        suffix = "_rewrite" if args.rewrite else ""
        path = settings.data_dir / "eval" / f"retrieval_eval{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(report.as_markdown())
        print(f"\nSaved to {path}")
    elif args.command == "ask":
        from cs336_rag import db, rag
        from cs336_rag.config import get_settings

        settings = get_settings()
        with db.connect(settings) as conn:
            result = rag.answer(settings, conn, args.question, variant=args.variant)
        print(result.answer)
        print("\nSources:")
        for index, chunk in enumerate(result.sources, start=1):
            print(f"  [{index}] {chunk.title} — {chunk.url}")
    elif args.command == "evaluate-prompts":
        from cs336_rag import db, rag
        from cs336_rag.config import get_settings
        from cs336_rag.embeddings import EmbeddingClient
        from cs336_rag.evals.answer_eval import evaluate_prompts, sample_questions
        from cs336_rag.evals.ground_truth import load_ground_truth
        from cs336_rag.llm import build_openai_client
        from cs336_rag.rag import PROMPT_VARIANTS

        settings = get_settings()
        entries = load_ground_truth(settings.data_dir / "ground_truth.json")
        questions = sample_questions([e.question for e in entries], args.sample, args.seed)
        embedder = EmbeddingClient(settings)
        gen_client = build_openai_client(settings, purpose="generate answers")
        judge_client = build_openai_client(settings, purpose="judge answers")
        with db.connect(settings) as conn:
            answer_report = evaluate_prompts(
                settings,
                questions=questions,
                variants=list(PROMPT_VARIANTS),
                retrieve=lambda q: rag.retrieve_context(settings, conn, q, embedder=embedder),
                gen_client=gen_client,
                judge_client=judge_client,
            )
        path = settings.data_dir / "eval" / "answer_eval.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(answer_report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(answer_report.as_markdown())
        print(f"\nBest variant: {answer_report.best_variant}\nSaved to {path}")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "cs336_rag.api:app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
