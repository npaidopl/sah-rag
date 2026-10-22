"""Tests for the common cross-baseline retrieval evaluation contract."""

from __future__ import annotations

import pytest

from evaluation import RetrievalEvaluationCase, RetrievalJudgment
from evaluation.contracts import (
    CommonRetrievalEvaluationRun,
    evaluate_retrieval_run,
    validate_evaluation_results,
)


def _case(query_id: str, retrieved: tuple[str, ...], judgments) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        query_id=query_id,
        query_text=query_id,
        retrieval_method="unit",
        evaluation_level="document",
        judgments=tuple(RetrievalJudgment(item_id=i, relevance=r) for i, r in judgments),
        retrieved_ids=retrieved,
        metadata={},
    )


def _run(*, cutoffs=(1,), expected=("q-1",)) -> CommonRetrievalEvaluationRun:
    return CommonRetrievalEvaluationRun(
        experiment_cell_id="experiment-cell-" + "a" * 64,
        dataset_id="synthetic",
        architecture_id="longrag",
        cutoffs=tuple(cutoffs),
        expected_query_ids=tuple(expected),
    )


def _eval(run, cases):
    qm, agg = evaluate_retrieval_run(run=run, cases=cases)
    validate_evaluation_results(
        run=run, cases=cases, query_metrics=qm, aggregate_metrics=agg
    )
    return qm, agg


# --- metric semantics ---

def test_perfect_top1_result():
    c = _case("q-1", ("a", "b"), [("a", 1), ("b", 0)])
    qm, _ = _eval(_run(), (c,))
    assert qm[0].hit_at_k == 1.0
    assert qm[0].precision_at_k == 1.0
    assert qm[0].reciprocal_rank == 1.0
    assert qm[0].ndcg_at_k == 1.0


def test_no_relevant_within_cutoff():
    c = _case("q-1", ("x", "a"), [("a", 1)])
    qm, _ = _eval(_run(), (c,))
    assert qm[0].hit_at_k == 0.0
    assert qm[0].reciprocal_rank == 0.0
    assert qm[0].precision_at_k == 0.0


def test_recall_two_relevant_cutoff_one():
    c = _case("q-1", ("a", "b"), [("a", 1), ("b", 1)])
    qm, _ = _eval(_run(), (c,))
    assert qm[0].recall_at_k == 0.5


def test_precision_uses_fixed_cutoff_denominator():
    c = _case("q-1", ("a",), [("a", 1)])
    qm, _ = _eval(_run(cutoffs=(5,)), (c,))
    assert qm[0].precision_at_k == pytest.approx(1.0 / 5.0)
    assert qm[0].recall_at_k == 1.0


def test_reciprocal_rank_respects_cutoff():
    c = _case("q-1", ("x", "a"), [("a", 1)])
    qm1, _ = _eval(_run(cutoffs=(1,)), (c,))
    qm2, _ = _eval(_run(cutoffs=(5,)), (c,))
    assert qm1[0].reciprocal_rank == 0.0
    assert qm2[0].reciprocal_rank == pytest.approx(0.5)


def test_graded_ndcg_perfect_order_equals_one():
    c = _case("q-1", ("a", "b", "c"), [("a", 3), ("b", 2), ("c", 1)])
    qm, _ = _eval(_run(cutoffs=(10,)), (c,))
    assert qm[0].ndcg_at_k == pytest.approx(1.0)


def test_graded_ndcg_penalizes_reversed_order():
    good = _case("q-1", ("a", "b", "c"), [("a", 3), ("b", 2), ("c", 1)])
    bad = _case("q-1", ("c", "b", "a"), [("a", 3), ("b", 2), ("c", 1)])
    qm_good, _ = _eval(_run(cutoffs=(10,)), (good,))
    qm_bad, _ = _eval(_run(cutoffs=(10,)), (bad,))
    assert qm_good[0].ndcg_at_k == pytest.approx(1.0)
    assert qm_bad[0].ndcg_at_k < qm_good[0].ndcg_at_k


def test_duplicate_retrieved_id_rejected():
    # The RetrievalEvaluationCase model itself refuses duplicate
    # retrieved_ids before the evaluator is ever invoked.
    with pytest.raises(ValueError, match="Retrieved IDs must be unique"):
        _case("q-1", ("a", "a"), [("a", 1)])


def test_duplicate_judgment_id_rejected():
    # Same: the model refuses duplicate judgment item_ids at
    # construction time.
    with pytest.raises(ValueError, match="duplicate item IDs"):
        _case("q-1", ("a",), [("a", 1), ("a", 0)])


def test_zero_relevant_case_rejected():
    c = _case("q-1", ("a",), [("a", 0)])
    with pytest.raises(ValueError, match="no relevant judgment"):
        _eval(_run(), (c,))


# --- run coverage ---

def test_missing_query_rejected():
    c = _case("q-1", ("a",), [("a", 1)])
    run = _run(expected=("q-1", "q-2"))
    with pytest.raises(ValueError, match="do not match expected"):
        evaluate_retrieval_run(run=run, cases=(c,))


def test_unknown_query_rejected():
    c = _case("q-2", ("a",), [("a", 1)])
    run = _run(expected=("q-1",))
    with pytest.raises(ValueError, match="do not match expected"):
        evaluate_retrieval_run(run=run, cases=(c,))


def test_duplicate_query_case_rejected():
    c1 = _case("q-1", ("a",), [("a", 1)])
    c2 = _case("q-1", ("b",), [("b", 1)])
    run = _run(expected=("q-1",))
    with pytest.raises(ValueError, match="duplicate query_id"):
        evaluate_retrieval_run(run=run, cases=(c1, c2))


def test_multi_cutoff_result_count_exact():
    c = _case("q-1", ("a",), [("a", 1)])
    run = _run(cutoffs=(1, 5, 10, 100), expected=("q-1",))
    qm, agg = evaluate_retrieval_run(run=run, cases=(c,))
    assert len(qm) == 4
    assert len(agg) == 4
    assert [a.cutoff for a in agg] == [1, 5, 10, 100]


def test_macro_aggregation_equal_query_weight():
    c1 = _case("q-1", ("a",), [("a", 1)])
    c2 = _case("q-2", ("x",), [("x", 1)])
    run = _run(cutoffs=(1,), expected=("q-1", "q-2"))
    qm, agg = evaluate_retrieval_run(run=run, cases=(c1, c2))
    assert len(qm) == 2
    assert agg[0].query_count == 2
    assert agg[0].mean_hit_at_k == 1.0
