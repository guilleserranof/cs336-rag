"""Unit tests for retrieval evaluation validation."""

from unittest.mock import MagicMock

import pytest

from cs336_rag.evals.ground_truth import GroundTruthEntry
from cs336_rag.evals.retrieval_eval import evaluate_retrieval


def test_limit_must_cover_hit_rate_10() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        evaluate_retrieval(
            settings=MagicMock(),
            conn=MagicMock(),
            entries=[GroundTruthEntry(question="q?", chunk_id="vid1:0")],
            methods=["text"],
            embedder=MagicMock(),
            limit=5,
        )
