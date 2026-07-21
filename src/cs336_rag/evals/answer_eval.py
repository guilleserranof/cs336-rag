"""Compare RAG prompt variants with an LLM judge.

For each evaluation question the context is retrieved once and shared
across variants, each variant generates an answer, and a separate judge
model (``Settings.judge_model``, different from the generator to reduce
self-preference bias) rates the answer for relevance, groundedness and
citation quality on a 1-5 scale. Variants are ranked by mean overall
score; the winner is wired in as ``Settings.rag_prompt_variant``.
"""

import logging
import random
from collections.abc import Callable
from datetime import UTC, datetime

from openai import OpenAI
from pydantic import BaseModel

from cs336_rag.config import Settings
from cs336_rag.evals.json_scan import scan_json
from cs336_rag.llm import retry_transient
from cs336_rag.models import Chunk
from cs336_rag.rag import EmptyAnswerError, format_context, generate_answer

logger = logging.getLogger(__name__)

_SCORE_MIN = 1
_SCORE_MAX = 5

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of answers to questions about the Stanford CS336 "
    "lecture series. Given a question, the numbered context passages the answer was "
    "supposed to use, and the answer, rate it on three axes from 1 to 5:\n"
    "- relevance: does the answer directly and completely address the question?\n"
    "- groundedness: is every claim supported by the context, with no invented facts?\n"
    "- citation: are the specific claims attributed to the numbered passages with "
    "inline markers like [1], [2]? An answer with no citations scores 1 here.\n"
    "Be discerning and use the full range; reserve 5 for an answer that genuinely "
    "could not be improved on that axis.\n"
    "Reply with ONLY a JSON object: "
    '{"relevance": <1-5>, "groundedness": <1-5>, "citation": <1-5>}.'
)

JUDGE_USER_TEMPLATE = "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"

# retrieve(question) -> context chunks; injected so the eval shares the app's retrieval
RetrieveFn = Callable[[str], list[Chunk]]


class JudgeScore(BaseModel):
    relevance: int
    groundedness: int
    citation: int

    @property
    def overall(self) -> float:
        return (self.relevance + self.groundedness + self.citation) / 3


class PromptResult(BaseModel):
    avg_relevance: float
    avg_groundedness: float
    avg_citation: float
    avg_overall: float
    questions: int


class AnswerReport(BaseModel):
    generated_at: datetime
    results: dict[str, PromptResult]

    @property
    def best_variant(self) -> str | None:
        scored = {name: r for name, r in self.results.items() if r.questions > 0}
        if not scored:
            return None
        return max(scored, key=lambda name: scored[name].avg_overall)

    def as_markdown(self) -> str:
        lines = [
            "| Variant | Relevance | Groundedness | Citation | Overall | n |",
            "|---|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {name} | {r.avg_relevance:.2f} | {r.avg_groundedness:.2f} "
            f"| {r.avg_citation:.2f} | {r.avg_overall:.2f} | {r.questions} |"
            for name, r in self.results.items()
        )
        return "\n".join(lines)


def sample_questions(questions: list[str], size: int, seed: int) -> list[str]:
    """Deterministic random sample of questions (capped at the population)."""
    rng = random.Random(seed)
    return rng.sample(questions, min(size, len(questions)))


def _clamp(value: int) -> int:
    return max(_SCORE_MIN, min(_SCORE_MAX, value))


def _score_from(data: object) -> JudgeScore | None:
    """Convert a decoded JSON object into a JudgeScore, or None to keep scanning."""
    if not isinstance(data, dict):
        return None
    if not {"relevance", "groundedness", "citation"} <= data.keys():
        return None
    try:
        return JudgeScore(
            relevance=_clamp(int(data["relevance"])),
            groundedness=_clamp(int(data["groundedness"])),
            citation=_clamp(int(data["citation"])),
        )
    except (ValueError, TypeError):
        return None


def parse_judge(raw: str) -> JudgeScore | None:
    """Extract judge scores from a model reply; None when unparseable.

    Scans for the first JSON object carrying all three score fields (so a
    preamble object does not abort the search); clamps scores into 1-5.
    """
    return scan_json(raw, "{", _score_from)


@retry_transient
def _create_judgement(client: OpenAI, model: str, question: str, context: str, answer: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question, context=context, answer=answer
                ),
            },
        ],
        temperature=0.0,
    )
    if not completion.choices:
        return ""
    return completion.choices[0].message.content or ""


def judge_answer(
    settings: Settings,
    question: str,
    chunks: list[Chunk],
    answer: str,
    client: OpenAI,
) -> JudgeScore | None:
    """Ask the judge model to rate one answer; None when unparseable."""
    raw = _create_judgement(client, settings.judge_model, question, format_context(chunks), answer)
    return parse_judge(raw)


def _aggregate(scores: list[JudgeScore]) -> PromptResult:
    if not scores:
        return PromptResult(
            avg_relevance=0.0,
            avg_groundedness=0.0,
            avg_citation=0.0,
            avg_overall=0.0,
            questions=0,
        )
    n = len(scores)
    return PromptResult(
        avg_relevance=sum(s.relevance for s in scores) / n,
        avg_groundedness=sum(s.groundedness for s in scores) / n,
        avg_citation=sum(s.citation for s in scores) / n,
        avg_overall=sum(s.overall for s in scores) / n,
        questions=n,
    )


def evaluate_prompts(
    settings: Settings,
    questions: list[str],
    variants: list[str],
    retrieve: RetrieveFn,
    gen_client: OpenAI,
    judge_client: OpenAI,
) -> AnswerReport:
    """Generate an answer per (question, variant) and judge each.

    Context is retrieved once per question and reused across variants so the
    only thing that differs between variants is the prompt.
    """
    if not questions:
        raise ValueError("No questions to evaluate")
    contexts = {question: retrieve(question) for question in questions}

    results: dict[str, PromptResult] = {}
    for variant in variants:
        scores: list[JudgeScore] = []
        for question in questions:
            chunks = contexts[question]
            try:
                answer = generate_answer(
                    settings, question, chunks, variant=variant, client=gen_client
                )
            except EmptyAnswerError:
                logger.warning("Empty generation for variant %s, question %r", variant, question)
                continue
            score = judge_answer(settings, question, chunks, answer.answer, judge_client)
            if score is None:
                logger.warning("Unjudgeable answer for variant %s, question %r", variant, question)
                continue
            scores.append(score)
        results[variant] = _aggregate(scores)
        logger.info(
            "%s: relevance=%.2f groundedness=%.2f citation=%.2f overall=%.2f (n=%d)",
            variant,
            results[variant].avg_relevance,
            results[variant].avg_groundedness,
            results[variant].avg_citation,
            results[variant].avg_overall,
            results[variant].questions,
        )
    return AnswerReport(generated_at=datetime.now(UTC), results=results)
