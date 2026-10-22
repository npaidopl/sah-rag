"""Common cross-baseline retrieval evaluation contract (Step 67).

This module wraps the existing ``evaluation`` primitives in a
multi-cutoff, run-level contract with deterministic persistence. It does
NOT replace ``evaluate_case`` or ``aggregate_metrics``. The naming
``CommonRetrievalEvaluationRun`` / ``CommonRetrievalEvaluationManifest``
avoids a collision with ``evaluation.RetrievalEvaluationRun``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation import (
    RetrievalEvaluationCase,
    aggregate_metrics,
    evaluate_case,
)

EVALUATION_CONTRACT_VERSION = "0.1.0"
DEFAULT_CUTOFFS: tuple[int, ...] = (1, 5, 10, 100)


class QueryCutoffMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_cell_id: str
    dataset_id: str
    architecture_id: str
    query_id: str
    cutoff: int = Field(gt=0)

    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float

    relevant_item_count: int = Field(ge=0)
    retrieved_item_count: int = Field(ge=0)
    retrieved_within_cutoff_count: int = Field(ge=0)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "experiment_cell_id",
        "dataset_id",
        "architecture_id",
        "query_id",
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator(
        "hit_at_k",
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
    )
    @classmethod
    def _unit(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ValueError("metric must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def _invariants(self) -> QueryCutoffMetrics:
        if self.retrieved_within_cutoff_count > self.cutoff:
            raise ValueError("retrieved_within_cutoff_count exceeds cutoff")
        if self.retrieved_within_cutoff_count > self.retrieved_item_count:
            raise ValueError(
                "retrieved_within_cutoff_count exceeds retrieved_item_count"
            )
        return self


class AggregateCutoffMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_cell_id: str
    dataset_id: str
    architecture_id: str
    cutoff: int = Field(gt=0)
    query_count: int = Field(gt=0)

    mean_hit_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "mean_hit_at_k",
        "mean_precision_at_k",
        "mean_recall_at_k",
        "mean_reciprocal_rank",
        "mean_ndcg_at_k",
    )
    @classmethod
    def _unit(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ValueError("mean must be in [0, 1]")
        return v


class CommonRetrievalEvaluationRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_cell_id: str
    dataset_id: str
    architecture_id: str

    cutoffs: tuple[int, ...]
    expected_query_ids: tuple[str, ...]

    retrieval_only: bool = True
    final_generation_enabled: bool = False
    answer_evaluation_enabled: bool = False
    llm_as_judge_enabled: bool = False

    evaluation_contract_version: str = EVALUATION_CONTRACT_VERSION
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_cell_id", "dataset_id", "architecture_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("cutoffs")
    @classmethod
    def _cutoffs(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("cutoffs must be non-empty")
        if any(c <= 0 for c in v):
            raise ValueError("cutoffs must be positive")
        if len(set(v)) != len(v):
            raise ValueError("cutoffs must be unique")
        if tuple(sorted(v)) != tuple(v):
            raise ValueError("cutoffs must be sorted")
        return v

    @field_validator("expected_query_ids")
    @classmethod
    def _queries(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("expected_query_ids must be non-empty")
        if any(not q for q in v):
            raise ValueError("expected_query_ids must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError("expected_query_ids must be unique")
        return v

    @model_validator(mode="after")
    def _flags(self) -> CommonRetrievalEvaluationRun:
        if self.retrieval_only is not True:
            raise ValueError("retrieval_only must be True")
        if self.final_generation_enabled is not False:
            raise ValueError("final_generation_enabled must be False")
        if self.answer_evaluation_enabled is not False:
            raise ValueError("answer_evaluation_enabled must be False")
        if self.llm_as_judge_enabled is not False:
            raise ValueError("llm_as_judge_enabled must be False")
        return self


class CommonRetrievalEvaluationManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_cell_id: str
    dataset_id: str
    architecture_id: str

    query_count: int = Field(ge=0)
    cutoffs: tuple[int, ...]

    case_count: int = Field(ge=0)
    query_metric_count: int = Field(ge=0)
    aggregate_metric_count: int = Field(ge=0)

    cases_path: str
    query_metrics_path: str
    aggregate_metrics_path: str

    cases_sha256: str
    query_metrics_sha256: str
    aggregate_metrics_sha256: str

    retrieval_only: bool = True
    final_generation_enabled: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _flags(self) -> CommonRetrievalEvaluationManifest:
        if self.retrieval_only is not True:
            raise ValueError("retrieval_only must be True")
        if self.final_generation_enabled is not False:
            raise ValueError("final_generation_enabled must be False")
        return self


def evaluate_retrieval_run(
    *,
    run: CommonRetrievalEvaluationRun,
    cases: Sequence[RetrievalEvaluationCase],
) -> tuple[
    tuple[QueryCutoffMetrics, ...],
    tuple[AggregateCutoffMetrics, ...],
]:
    """Evaluate a multi-cutoff retrieval run via the existing primitives."""
    case_list = list(cases)
    case_ids = [c.query_id for c in case_list]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate query_id in cases")

    expected = set(run.expected_query_ids)
    got = set(case_ids)
    if got != expected:
        missing = sorted(expected - got)
        unknown = sorted(got - expected)
        raise ValueError(
            f"case query_ids do not match expected: "
            f"missing={missing} unknown={unknown}"
        )

    for c in case_list:
        if len(set(c.retrieved_ids)) != len(c.retrieved_ids):
            raise ValueError(f"duplicate retrieved_ids in {c.query_id}")
        jids = [j.item_id for j in c.judgments]
        if len(set(jids)) != len(jids):
            raise ValueError(f"duplicate judgment ids in {c.query_id}")
        if not any(j.relevance > 0 for j in c.judgments):
            raise ValueError(f"no relevant judgment in {c.query_id}")

    ordered_cases = sorted(case_list, key=lambda c: c.query_id)

    per_case: dict[tuple[str, int], Any] = {}
    for c in ordered_cases:
        for cutoff in run.cutoffs:
            per_case[(c.query_id, cutoff)] = evaluate_case(c, cutoff=cutoff)

    query_metrics: list[QueryCutoffMetrics] = []
    for c in ordered_cases:
        for cutoff in run.cutoffs:
            m = per_case[(c.query_id, cutoff)]
            query_metrics.append(
                QueryCutoffMetrics(
                    experiment_cell_id=run.experiment_cell_id,
                    dataset_id=run.dataset_id,
                    architecture_id=run.architecture_id,
                    query_id=c.query_id,
                    cutoff=cutoff,
                    hit_at_k=float(m.hit_at_k),
                    precision_at_k=float(m.precision_at_k),
                    recall_at_k=float(m.recall_at_k),
                    reciprocal_rank=float(m.reciprocal_rank),
                    ndcg_at_k=float(m.ndcg_at_k),
                    relevant_item_count=int(m.relevant_count),
                    retrieved_item_count=int(m.retrieved_count),
                    retrieved_within_cutoff_count=min(
                        cutoff, int(m.retrieved_count)
                    ),
                )
            )

    aggregates: list[AggregateCutoffMetrics] = []
    for cutoff in run.cutoffs:
        per_cutoff = [per_case[(c.query_id, cutoff)] for c in ordered_cases]
        agg = aggregate_metrics(per_cutoff)
        aggregates.append(
            AggregateCutoffMetrics(
                experiment_cell_id=run.experiment_cell_id,
                dataset_id=run.dataset_id,
                architecture_id=run.architecture_id,
                cutoff=cutoff,
                query_count=len(per_cutoff),
                mean_hit_at_k=float(agg.mean_hit_at_k),
                mean_precision_at_k=float(agg.mean_precision_at_k),
                mean_recall_at_k=float(agg.mean_recall_at_k),
                mean_reciprocal_rank=float(agg.mean_reciprocal_rank),
                mean_ndcg_at_k=float(agg.mean_ndcg_at_k),
            )
        )

    return tuple(query_metrics), tuple(aggregates)


def validate_evaluation_results(
    *,
    run: CommonRetrievalEvaluationRun,
    cases: Sequence[RetrievalEvaluationCase],
    query_metrics: Sequence[QueryCutoffMetrics],
    aggregate_metrics: Sequence[AggregateCutoffMetrics],
) -> None:
    n_queries = len(run.expected_query_ids)
    n_cutoffs = len(run.cutoffs)

    if len(query_metrics) != n_queries * n_cutoffs:
        raise ValueError("query_metric_count != query_count * cutoff_count")
    if len(aggregate_metrics) != n_cutoffs:
        raise ValueError("aggregate_metric_count != cutoff_count")

    seen: set[tuple[str, int]] = set()
    for m in query_metrics:
        if (m.query_id, m.cutoff) in seen:
            raise ValueError("duplicate (query_id, cutoff) in query_metrics")
        seen.add((m.query_id, m.cutoff))
        if m.cutoff not in run.cutoffs:
            raise ValueError("query metric cutoff not in run.cutoffs")
        if m.query_id not in run.expected_query_ids:
            raise ValueError("query metric query_id not expected")

    seen_cutoffs: set[int] = set()
    for a in aggregate_metrics:
        if a.cutoff in seen_cutoffs:
            raise ValueError("duplicate cutoff in aggregate_metrics")
        seen_cutoffs.add(a.cutoff)
        if a.cutoff not in run.cutoffs:
            raise ValueError("aggregate cutoff not in run.cutoffs")

    if len(cases) != n_queries:
        raise ValueError("cases count != expected_query_count")
