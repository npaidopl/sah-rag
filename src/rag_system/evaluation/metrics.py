"""Standard ranked-retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rag_system.evaluation.models import (
    RetrievalAggregateMetrics,
    RetrievalEvaluationCase,
    RetrievalMetrics,
)


def recall_at_k(
    *,
    retrieved_ids: Sequence[str],
    relevance_by_id: Mapping[str, int],
    cutoff: int,
) -> float:
    """Calculate binary Recall@k."""
    _validate_cutoff(cutoff)

    relevant_ids = {
        item_id
        for item_id, relevance in relevance_by_id.items()
        if relevance > 0
    }

    if not relevant_ids:
        return 0.0

    retrieved_relevant = relevant_ids.intersection(
        retrieved_ids[:cutoff]
    )

    return len(retrieved_relevant) / len(relevant_ids)


def precision_at_k(
    *,
    retrieved_ids: Sequence[str],
    relevance_by_id: Mapping[str, int],
    cutoff: int,
) -> float:
    """Calculate Precision@k using the fixed cutoff."""
    _validate_cutoff(cutoff)

    relevant_retrieved = sum(
        relevance_by_id.get(item_id, 0) > 0
        for item_id in retrieved_ids[:cutoff]
    )

    return relevant_retrieved / cutoff


def reciprocal_rank(
    *,
    retrieved_ids: Sequence[str],
    relevance_by_id: Mapping[str, int],
    cutoff: int,
) -> float:
    """Calculate reciprocal rank of the first relevant item."""
    _validate_cutoff(cutoff)

    for rank, item_id in enumerate(
        retrieved_ids[:cutoff],
        start=1,
    ):
        if relevance_by_id.get(item_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    *,
    retrieved_ids: Sequence[str],
    relevance_by_id: Mapping[str, int],
    cutoff: int,
) -> float:
    """Calculate graded nDCG@k with exponential gain."""
    _validate_cutoff(cutoff)

    dcg = _discounted_cumulative_gain(
        [
            relevance_by_id.get(item_id, 0)
            for item_id in retrieved_ids[:cutoff]
        ]
    )

    ideal_relevances = sorted(
        relevance_by_id.values(),
        reverse=True,
    )[:cutoff]

    ideal_dcg = _discounted_cumulative_gain(
        ideal_relevances
    )

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg


def evaluate_case(
    case: RetrievalEvaluationCase,
    *,
    cutoff: int,
) -> RetrievalMetrics:
    """Evaluate one ranked retrieval case."""
    _validate_cutoff(cutoff)

    relevance_by_id = {
        judgment.item_id: judgment.relevance
        for judgment in case.judgments
    }

    relevant_ids = {
        item_id
        for item_id, relevance in relevance_by_id.items()
        if relevance > 0
    }

    top_ids = case.retrieved_ids[:cutoff]

    relevant_retrieved_count = sum(
        item_id in relevant_ids
        for item_id in top_ids
    )

    recall = recall_at_k(
        retrieved_ids=case.retrieved_ids,
        relevance_by_id=relevance_by_id,
        cutoff=cutoff,
    )

    return RetrievalMetrics(
        query_id=case.query_id,
        retrieval_method=case.retrieval_method,
        evaluation_level=case.evaluation_level,
        cutoff=cutoff,
        relevant_count=len(relevant_ids),
        retrieved_count=len(top_ids),
        relevant_retrieved_count=(
            relevant_retrieved_count
        ),
        recall_at_k=recall,
        precision_at_k=precision_at_k(
            retrieved_ids=case.retrieved_ids,
            relevance_by_id=relevance_by_id,
            cutoff=cutoff,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_ids=case.retrieved_ids,
            relevance_by_id=relevance_by_id,
            cutoff=cutoff,
        ),
        ndcg_at_k=ndcg_at_k(
            retrieved_ids=case.retrieved_ids,
            relevance_by_id=relevance_by_id,
            cutoff=cutoff,
        ),
        hit_at_k=float(
            relevant_retrieved_count > 0
        ),
    )


def aggregate_metrics(
    metrics: Sequence[RetrievalMetrics],
) -> RetrievalAggregateMetrics:
    """Calculate macro averages for compatible cases."""
    if not metrics:
        raise ValueError(
            "At least one metric record is required."
        )

    first = metrics[0]

    for metric in metrics[1:]:
        if (
            metric.retrieval_method
            != first.retrieval_method
        ):
            raise ValueError(
                "Cannot aggregate different retrieval methods."
            )

        if (
            metric.evaluation_level
            != first.evaluation_level
        ):
            raise ValueError(
                "Cannot aggregate different evaluation levels."
            )

        if metric.cutoff != first.cutoff:
            raise ValueError(
                "Cannot aggregate different cutoffs."
            )

    query_count = len(metrics)

    return RetrievalAggregateMetrics(
        retrieval_method=first.retrieval_method,
        evaluation_level=first.evaluation_level,
        cutoff=first.cutoff,
        query_count=query_count,
        mean_recall_at_k=sum(
            metric.recall_at_k
            for metric in metrics
        )
        / query_count,
        mean_precision_at_k=sum(
            metric.precision_at_k
            for metric in metrics
        )
        / query_count,
        mean_reciprocal_rank=sum(
            metric.reciprocal_rank
            for metric in metrics
        )
        / query_count,
        mean_ndcg_at_k=sum(
            metric.ndcg_at_k
            for metric in metrics
        )
        / query_count,
        mean_hit_at_k=sum(
            metric.hit_at_k
            for metric in metrics
        )
        / query_count,
    )


def _discounted_cumulative_gain(
    relevances: Sequence[int],
) -> float:
    """Calculate DCG with gain 2^relevance - 1."""
    return float(
        sum(
            (
                (2**relevance) - 1
            )
            / math.log2(rank + 1)
            for rank, relevance in enumerate(
                relevances,
                start=1,
            )
        )
    )


def _validate_cutoff(cutoff: int) -> None:
    """Require a positive metric cutoff."""
    if cutoff < 1:
        raise ValueError(
            "Metric cutoff must be positive."
        )
