"""Tests for ranking metrics (pure functions)."""

import pytest

from cs336_rag.evals.metrics import hit_rate_at, mrr, rank_of


class TestRankOf:
    def test_first_position_is_rank_one(self) -> None:
        assert rank_of(["a", "b", "c"], "a") == 1

    def test_later_position(self) -> None:
        assert rank_of(["a", "b", "c"], "c") == 3

    def test_missing_returns_none(self) -> None:
        assert rank_of(["a", "b"], "z") is None

    def test_empty_list(self) -> None:
        assert rank_of([], "a") is None


class TestHitRateAt:
    def test_all_hits_within_k(self) -> None:
        assert hit_rate_at([1, 2, 3], k=5) == 1.0

    def test_misses_and_out_of_range(self) -> None:
        # rank 6 is beyond k=5, None never retrieved
        assert hit_rate_at([1, 6, None, 4], k=5) == pytest.approx(0.5)

    def test_empty_ranks_is_zero(self) -> None:
        assert hit_rate_at([], k=5) == 0.0


class TestMrr:
    def test_perfect_retrieval(self) -> None:
        assert mrr([1, 1, 1]) == 1.0

    def test_mixed_ranks(self) -> None:
        # 1/1, 1/2, miss -> (1 + 0.5 + 0) / 3
        assert mrr([1, 2, None]) == pytest.approx(0.5)

    def test_empty_is_zero(self) -> None:
        assert mrr([]) == 0.0
