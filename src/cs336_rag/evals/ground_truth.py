"""Generate an evaluation dataset of questions from knowledge-base chunks.

For a random (seeded, reproducible) sample of chunks, the chat model
writes questions that each chunk answers. The originating ``chunk_id`` is
the relevance label for retrieval evaluation. The generated dataset is
committed to ``data/ground_truth.json`` so evaluations are reproducible
without re-paying the generation cost.
"""

import json
import logging
import random
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from cs336_rag.llm import retry_transient
from cs336_rag.models import Chunk

logger = logging.getLogger(__name__)

GENERATION_SYSTEM_PROMPT = (
    "You create evaluation questions for a retrieval system over the Stanford "
    "CS336 'Language Modeling from Scratch' lecture transcripts."
)

GENERATION_USER_TEMPLATE = (
    "Passage from {title}:\n\n{content}\n\n"
    "Write {n} diverse, self-contained questions that this passage answers. "
    "Phrase them the way a student studying the course would ask, using the "
    "topic's own terminology. Never refer to 'the passage', 'the lecture' or "
    "'the speaker'. Reply with a JSON array of exactly {n} strings."
)


class GroundTruthEntry(BaseModel):
    question: str
    chunk_id: str


def parse_questions(raw: str) -> list[str]:
    """Extract a JSON array of question strings from a model reply.

    Scans for the first ``[`` that begins a valid JSON array, tolerating code
    fences, surrounding prose and stray brackets elsewhere in the text.
    Non-string items are dropped; returns [] when nothing parses.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "[":
            continue
        try:
            data, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [item for item in data if isinstance(item, str) and item.strip()]
    return []


def sample_chunks(chunks: list[Chunk], size: int, seed: int) -> list[Chunk]:
    """Deterministic random sample (capped at the population size)."""
    rng = random.Random(seed)
    return rng.sample(chunks, min(size, len(chunks)))


@retry_transient
def _ask_for_questions(client: OpenAI, model: str, chunk: Chunk, per_chunk: int) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GENERATION_USER_TEMPLATE.format(
                    title=chunk.title, content=chunk.content, n=per_chunk
                ),
            },
        ],
        temperature=0.7,
    )
    if not completion.choices:
        return ""
    return completion.choices[0].message.content or ""


def generate_ground_truth(
    chunks: list[Chunk], client: OpenAI, model: str, per_chunk: int = 2
) -> list[GroundTruthEntry]:
    """Generate ``per_chunk`` questions for every chunk; skip unparseable replies."""
    entries: list[GroundTruthEntry] = []
    for i, chunk in enumerate(chunks, start=1):
        raw = _ask_for_questions(client, model, chunk, per_chunk)
        questions = parse_questions(raw)[:per_chunk]
        if not questions:
            logger.warning("[%d/%d] no parseable questions for %s", i, len(chunks), chunk.id)
            continue
        entries.extend(
            GroundTruthEntry(question=question, chunk_id=chunk.id) for question in questions
        )
        if i % 25 == 0:
            logger.info("[%d/%d] generated %d questions so far", i, len(chunks), len(entries))
    return entries


def save_ground_truth(entries: list[GroundTruthEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.model_dump() for entry in entries]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_ground_truth(path: Path) -> list[GroundTruthEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GroundTruthEntry.model_validate(item) for item in data]
