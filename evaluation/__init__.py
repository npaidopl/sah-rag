"""Retrieval evaluation models, metrics, identity, and pipelines."""

from __future__ import annotations

from rag_system.evaluation.clerk import (
    ClerkQuery,
    ClerkRetrievalDataset,
    aggregate_clerk_document_chunks,
    build_clerk_evaluation_case,
    load_clerk_retrieval_dataset,
    rank_unique_documents,
)
from rag_system.evaluation.identity import (
    build_evaluation_run_id,
)
from rag_system.evaluation.metrics import (
    aggregate_metrics,
    evaluate_case,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_system.evaluation.models import (
    RetrievalAggregateMetrics,
    RetrievalEvaluationCase,
    RetrievalEvaluationRun,
    RetrievalJudgment,
    RetrievalMetrics,
)
from rag_system.evaluation.pipeline import (
    EvaluationArtifactPaths,
    EvaluationWriteSummary,
    resolve_evaluation_paths,
    write_retrieval_evaluation,
)
from rag_system.evaluation.statistics import (
    PairedBootstrapResult,
    paired_bootstrap_difference,
)

__all__ = [
    "PairedBootstrapResult",
    "paired_bootstrap_difference",
    "ClerkQuery",
    "ClerkRetrievalDataset",
    "EvaluationArtifactPaths",
    "EvaluationWriteSummary",
    "RetrievalAggregateMetrics",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationRun",
    "RetrievalJudgment",
    "RetrievalMetrics",
    "aggregate_metrics",
    "build_clerk_evaluation_case",
    "aggregate_clerk_document_chunks",
    "build_evaluation_run_id",
    "evaluate_case",
    "load_clerk_retrieval_dataset",
    "ndcg_at_k",
    "precision_at_k",
    "rank_unique_documents",
    "recall_at_k",
    "reciprocal_rank",
    "resolve_evaluation_paths",
    "write_retrieval_evaluation",
]
