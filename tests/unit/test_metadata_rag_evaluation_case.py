"""Tests for rag_system.baselines.metadata_rag_evaluation_case."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_system.baselines.metadata_rag_evaluation_case import (
    EVALUATION_CASE_VERSION,
    MetadataRAGEvaluationCase,
    MetadataRAGEvaluationCaseError,
    build_evaluation_case,
)


def test_build_case_shape() -> None:
    case = build_evaluation_case(
        query_id="q-1",
        ranked_doc_ids=["d-1", "d-2", "d-3"],
        relevant_doc_ids=["d-2"],
    )
    assert case.query_id == "q-1"
    assert case.ranked_doc_ids == ("d-1", "d-2", "d-3")
    assert case.relevant_doc_ids == ("d-2",)
    assert case.metadata == {}


def test_build_case_with_empty_ranked_list() -> None:
    case = build_evaluation_case(
        query_id="q",
        ranked_doc_ids=[],
        relevant_doc_ids=["d-1"],
    )
    assert case.ranked_doc_ids == ()


def test_build_case_with_empty_relevant_list() -> None:
    case = build_evaluation_case(
        query_id="q",
        ranked_doc_ids=["d-1"],
        relevant_doc_ids=[],
    )
    assert case.relevant_doc_ids == ()


def test_build_case_preserves_ranking_order() -> None:
    case = build_evaluation_case(
        query_id="q",
        ranked_doc_ids=["z", "a", "m"],
        relevant_doc_ids=["a"],
    )
    assert case.ranked_doc_ids == ("z", "a", "m")


def test_build_case_rejects_empty_query_id() -> None:
    with pytest.raises(
        MetadataRAGEvaluationCaseError, match="query_id"
    ):
        build_evaluation_case(
            query_id="",
            ranked_doc_ids=["d"],
            relevant_doc_ids=["d"],
        )


def test_case_model_rejects_empty_doc_id_in_ranked() -> None:
    with pytest.raises(ValidationError):
        MetadataRAGEvaluationCase(
            query_id="q",
            ranked_doc_ids=("d", ""),
            relevant_doc_ids=("d",),
        )


def test_case_model_rejects_empty_doc_id_in_relevant() -> None:
    with pytest.raises(ValidationError):
        MetadataRAGEvaluationCase(
            query_id="q",
            ranked_doc_ids=("d",),
            relevant_doc_ids=("",),
        )


def test_case_model_rejects_duplicate_ranked_doc_ids() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        MetadataRAGEvaluationCase(
            query_id="q",
            ranked_doc_ids=("d", "d"),
            relevant_doc_ids=("d",),
        )


def test_case_model_allows_duplicate_relevant_doc_ids() -> None:
    # Ground truth could conceivably repeat a relevant doc_id.
    case = MetadataRAGEvaluationCase(
        query_id="q",
        ranked_doc_ids=("d",),
        relevant_doc_ids=("d", "d"),
    )
    assert case.relevant_doc_ids == ("d", "d")


def test_case_is_frozen() -> None:
    case = build_evaluation_case(
        query_id="q",
        ranked_doc_ids=["d"],
        relevant_doc_ids=["d"],
    )
    with pytest.raises(ValidationError):
        case.query_id = "changed"


def test_evaluation_case_version_is_exposed() -> None:
    assert EVALUATION_CASE_VERSION == "0.1.0"
