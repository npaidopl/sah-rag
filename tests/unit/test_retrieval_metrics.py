"""Tests for ranked-retrieval metrics."""

from __future__ import annotations

import math

import pytest

from evaluation import (
    RetrievalEvaluationCase,
    RetrievalJudgment,
    aggregate_metrics,
    evaluate_case,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def judgments() -> dict[str, int]:
    """Return graded sample judgments."""
    return {
        "a": 2,
        "b": 1,
        "c": 0,
    }


def test_recall_at_k() -> None:
    """Recall should measure recovered relevant items."""
    value = recall_at_k(
        retrieved_ids=("a", "c", "x"),
        relevance_by_id=judgments(),
        cutoff=2,
    )

    assert value == 0.5


def test_precision_at_k_uses_fixed_cutoff() -> None:
    """Precision should divide by the requested cutoff."""
    value = precision_at_k(
        retrieved_ids=("a",),
        relevance_by_id=judgments(),
        cutoff=2,
    )

    assert value == 0.5


def test_reciprocal_rank() -> None:
    """The first relevant rank should determine RR."""
    value = reciprocal_rank(
        retrieved_ids=("x", "b", "a"),
        relevance_by_id=judgments(),
        cutoff=3,
    )

    assert value == 0.5


def test_ndcg_is_one_for_ideal_order() -> None:
    """An ideal graded ranking should have nDCG 1."""
    value = ndcg_at_k(
        retrieved_ids=("a", "b", "c"),
        relevance_by_id=judgments(),
        cutoff=3,
    )

    assert math.isclose(value, 1.0)


def test_ndcg_penalizes_inverted_order() -> None:
    """Lower-grade-first order should reduce nDCG."""
    value = ndcg_at_k(
        retrieved_ids=("b", "a", "c"),
        relevance_by_id=judgments(),
        cutoff=3,
    )

    assert 0.0 < value < 1.0


def test_no_relevant_items_returns_zero() -> None:
    """Queries without positives should return zero metrics."""
    relevance = {
        "a": 0,
        "b": 0,
    }

    assert recall_at_k(
        retrieved_ids=("a", "b"),
        relevance_by_id=relevance,
        cutoff=2,
    ) == 0.0

    assert reciprocal_rank(
        retrieved_ids=("a", "b"),
        relevance_by_id=relevance,
        cutoff=2,
    ) == 0.0

    assert ndcg_at_k(
        retrieved_ids=("a", "b"),
        relevance_by_id=relevance,
        cutoff=2,
    ) == 0.0


def test_evaluate_case() -> None:
    """A complete case should produce aligned metrics."""
    case = RetrievalEvaluationCase(
        query_id="query-001",
        query_text="sample query",
        retrieval_method="hybrid_rrf",
        evaluation_level="chunk",
        judgments=(
            RetrievalJudgment(
                item_id="a",
                relevance=2,
            ),
            RetrievalJudgment(
                item_id="b",
                relevance=1,
            ),
        ),
        retrieved_ids=("a", "x", "b"),
    )

    metrics = evaluate_case(
        case,
        cutoff=3,
    )

    assert metrics.relevant_count == 2
    assert metrics.relevant_retrieved_count == 2
    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.hit_at_k == 1.0


def test_aggregate_metrics() -> None:
    """Compatible cases should be macro-averaged."""
    cases = (
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="first",
            retrieval_method="dense",
            evaluation_level="chunk",
            judgments=(
                RetrievalJudgment(
                    item_id="a",
                    relevance=1,
                ),
            ),
            retrieved_ids=("a",),
        ),
        RetrievalEvaluationCase(
            query_id="query-002",
            query_text="second",
            retrieval_method="dense",
            evaluation_level="chunk",
            judgments=(
                RetrievalJudgment(
                    item_id="b",
                    relevance=1,
                ),
            ),
            retrieved_ids=("x",),
        ),
    )

    metrics = tuple(
        evaluate_case(case, cutoff=1)
        for case in cases
    )

    aggregate = aggregate_metrics(metrics)

    assert aggregate.query_count == 2
    assert aggregate.mean_recall_at_k == 0.5
    assert aggregate.mean_reciprocal_rank == 0.5
    assert aggregate.mean_hit_at_k == 0.5


def test_aggregate_rejects_mixed_methods() -> None:
    """Different retrieval methods cannot be mixed."""
    first = evaluate_case(
        RetrievalEvaluationCase(
            query_id="query-001",
            query_text="first",
            retrieval_method="dense",
            evaluation_level="chunk",
            judgments=(
                RetrievalJudgment(
                    item_id="a",
                    relevance=1,
                ),
            ),
            retrieved_ids=("a",),
        ),
        cutoff=1,
    )

    second = first.model_copy(
        update={
            "query_id": "query-002",
            "retrieval_method": "sparse",
        }
    )

    with pytest.raises(
        ValueError,
        match="different retrieval methods",
    ):
        aggregate_metrics((first, second))
