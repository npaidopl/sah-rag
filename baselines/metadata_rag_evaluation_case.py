"""Evaluation-case construction for the Metadata-RAG comparator.

The evaluation case pairs a query's ranked document list with the
canonical ground-truth relevant-document list. It is the ONLY object
in the Metadata-RAG pipeline that is allowed to see ground truth, and
it is constructed strictly after retrieval completes, by a process
that is independent of the retriever.

This module performs no metric computation. It builds the case
object only. Metric computation lives in the shared evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

EVALUATION_CASE_VERSION: Final[str] = "0.1.0"


class MetadataRAGEvaluationCaseError(ValueError):
    """Raised when evaluation-case construction violates its contract."""


class MetadataRAGBaseModel(BaseModel):
    """Strict, frozen base model for evaluation-case artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )


class MetadataRAGEvaluationCase(MetadataRAGBaseModel):
    """One retrieval-evaluation case.

    Fields:
        query_id: canonical query identifier.
        ranked_doc_ids: the ranked canonical document list produced
            by the retriever, in rank order.
        relevant_doc_ids: the canonical ground-truth relevant list.
        metadata: optional free-form metadata for downstream use.
    """

    query_id: str = Field(min_length=1)
    ranked_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_shape(self) -> MetadataRAGEvaluationCase:
        if not self.query_id.strip():
            raise ValueError("query_id cannot be empty")
        if any(not doc.strip() for doc in self.ranked_doc_ids):
            raise ValueError(
                "ranked_doc_ids must not contain empty identifiers"
            )
        if any(not doc.strip() for doc in self.relevant_doc_ids):
            raise ValueError(
                "relevant_doc_ids must not contain empty identifiers"
            )
        if len(set(self.ranked_doc_ids)) != len(self.ranked_doc_ids):
            raise ValueError(
                "ranked_doc_ids must not contain duplicates"
            )
        return self


def build_evaluation_case(
    *,
    query_id: str,
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
) -> MetadataRAGEvaluationCase:
    """Build one evaluation case.

    Callers must obtain ``ranked_doc_ids`` from the retriever output
    and ``relevant_doc_ids`` from the canonical ground truth. This
    function performs no retrieval and does not modify either list.
    """
    if not query_id.strip():
        raise MetadataRAGEvaluationCaseError(
            "query_id cannot be empty"
        )
    return MetadataRAGEvaluationCase(
        query_id=query_id,
        ranked_doc_ids=tuple(ranked_doc_ids),
        relevant_doc_ids=tuple(relevant_doc_ids),
        metadata={},
    )
