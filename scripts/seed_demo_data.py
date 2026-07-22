"""Seed synthetic telemetry so the Grafana dashboard has something to show.

This is a development/demo helper — it inserts fake conversations and
feedback spread over the last few days. It never touches the knowledge
base. Run with:

    uv run python scripts/seed_demo_data.py --count 400
"""

import argparse
import random
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from cs336_rag import db
from cs336_rag.config import get_settings

QUESTIONS = [
    "What is byte pair encoding and why is it used?",
    "How does FlashAttention reduce memory usage?",
    "Why is prenorm preferred over postnorm?",
    "What are scaling laws used for?",
    "How does rotary positional embedding work?",
    "What is the difference between data and model parallelism?",
    "Why do we use mixed-precision training?",
    "What is RLHF and how does it shape model behavior?",
]
VARIANTS = ["grounded", "grounded", "grounded", "tutor", "baseline"]  # grounded is default
METHODS = ["vector", "vector", "vector", "hybrid", "text"]


def seed(count: int, days: int, seed: int) -> None:
    rng = random.Random(seed)
    settings = get_settings()
    now = datetime.now(UTC)

    with db.connect(settings) as conn:
        db.init_app_schema(conn)
        for _ in range(count):
            conversation_id = uuid4()
            created = now - timedelta(seconds=rng.uniform(0, days * 24 * 3600))
            retrieval_ms = rng.gauss(1100, 300)
            generation_ms = rng.gauss(9000, 3500)
            prompt_tokens = rng.randint(900, 3000)
            completion_tokens = rng.randint(300, 1200)
            conn.execute(
                """
                INSERT INTO conversations (
                    id, question, answer, variant, retrieval_method, source_ids,
                    num_sources, retrieval_ms, generation_ms, total_ms,
                    prompt_tokens, completion_tokens, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    rng.choice(QUESTIONS),
                    "Synthetic demo answer with a citation [1].",
                    rng.choice(VARIANTS),
                    rng.choice(METHODS),
                    ["JuoVZkPBiKk:3", "JuoVZkPBiKk:4"],
                    rng.randint(3, 5),
                    max(retrieval_ms, 50),
                    max(generation_ms, 500),
                    max(retrieval_ms + generation_ms, 600),
                    prompt_tokens,
                    completion_tokens,
                    created,
                ),
            )
            # ~40% of conversations get a vote, ~78% of those positive
            if rng.random() < 0.4:
                rating = 1 if rng.random() < 0.78 else -1
                conn.execute(
                    "INSERT INTO feedback (conversation_id, rating, created_at) "
                    "VALUES (%s, %s, %s)",
                    (conversation_id, rating, created + timedelta(seconds=rng.uniform(5, 120))),
                )
        conn.commit()
    print(f"Seeded {count} synthetic conversations over the last {days} days.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=400, help="Conversations to insert.")
    parser.add_argument("--days", type=int, default=7, help="Spread over this many past days.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed.")
    args = parser.parse_args()
    seed(args.count, args.days, args.seed)


if __name__ == "__main__":
    main()
