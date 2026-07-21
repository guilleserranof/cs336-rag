"""Ranking metrics for retrieval evaluation.

Each ground-truth question knows the chunk it was generated from; a
retrieval method is scored by where that chunk lands in its ranking.
``rank`` values are 1-based, ``None`` meaning the chunk was not retrieved.
"""


def rank_of(retrieved_ids: list[str], relevant_id: str) -> int | None:
    """1-based rank of the relevant id in the retrieved list, or None."""
    try:
        return retrieved_ids.index(relevant_id) + 1
    except ValueError:
        return None


def hit_rate_at(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose relevant chunk appears in the top k."""
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank; misses contribute 0."""
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)
