"""Tests for canonical Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_system.canonical.models import (
    CanonicalDocument,
    CanonicalExample,
    GroundTruthType,
)


def make_example() -> dict[str, object]:
    """Return a valid canonical example dictionary."""

    return {
        "query": {
            "query_id": "query-001",
            "dataset": "example_dataset",
            "question": "What is the answer?",
            "task_type": "retrieval",
            "split": "test",
            "doc_ids": ["doc-001"],
            "metadata": {},
        },
        "ground_truth": {
            "ground_truth_type": "free_text",
            "dataset": "example_dataset",
            "query_id": "query-001",
            "answers": ["Example answer"],
            "relevant_ids": [],
            "choice": None,
            "spans": [],
            "metadata": {},
        },
        "supplied_context": None,
        "options": None,
        "metadata": {},
    }


def test_canonical_document_accepts_valid_record() -> None:
    """A normal canonical document should validate."""

    document = CanonicalDocument.model_validate(
        {
            "doc_id": "doc-001",
            "dataset": "example_dataset",
            "text_raw": "Raw text.",
            "text_normalized": "Normalized text.",
            "title": "Example",
            "source_path": "source.jsonl",
            "source_row": 1,
            "metadata": {"language": "en"},
        }
    )

    assert document.doc_id == "doc-001"
    assert document.source_row == 1


def test_canonical_document_rejects_empty_doc_id() -> None:
    """An empty canonical document ID should be rejected."""

    with pytest.raises(ValidationError):
        CanonicalDocument.model_validate(
            {
                "doc_id": "",
                "dataset": "example_dataset",
                "text_raw": "Raw text.",
                "text_normalized": "Normalized text.",
            }
        )


def test_canonical_example_accepts_known_ground_truth() -> None:
    """A valid example should produce nested typed models."""

    example = CanonicalExample.model_validate(make_example())

    assert example.query.query_id == "query-001"
    assert (
        example.ground_truth.ground_truth_type
        is GroundTruthType.FREE_TEXT
    )


def test_canonical_example_rejects_mismatched_query_id() -> None:
    """Query and ground-truth identity must agree."""

    raw_example = make_example()
    ground_truth = raw_example["ground_truth"]

    assert isinstance(ground_truth, dict)
    ground_truth["query_id"] = "different-query"

    with pytest.raises(ValidationError):
        CanonicalExample.model_validate(raw_example)


def test_canonical_example_rejects_unknown_ground_truth_type() -> None:
    """Unknown ground-truth families should be explicit failures."""

    raw_example = make_example()
    ground_truth = raw_example["ground_truth"]

    assert isinstance(ground_truth, dict)
    ground_truth["ground_truth_type"] = "unknown_type"

    with pytest.raises(ValidationError):
        CanonicalExample.model_validate(raw_example)


def test_extra_fields_are_preserved() -> None:
    """Future canonical fields should not be discarded."""

    raw_example = make_example()
    raw_example["future_field"] = {"enabled": True}

    example = CanonicalExample.model_validate(raw_example)
    dumped = example.model_dump()

    assert dumped["future_field"] == {"enabled": True}