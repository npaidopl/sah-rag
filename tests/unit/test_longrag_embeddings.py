"""Tests for typed LongRAG embedding artifacts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from baselines import (
    LongUnitEmbeddingRecord,
)


def make_record(**overrides: object) -> LongUnitEmbeddingRecord:
    """Create one valid embedding record."""
    values: dict[str, object] = {
        "long_unit_id": "longunit-clerk-abc",
        "condition_id": "LONGRAG-ADJ-4096",
        "dataset_id": "clerk",
        "construction_policy": "adjacent_documents",
        "target_token_size": 4096,
        "maximum_model_input_tokens": 4098,
        "unit_order": 0,
        "rendered_token_count": 4000,
        "special_token_count": 2,
        "actual_model_token_count": 4002,
        "was_truncated": False,
        "member_doc_ids": ("doc-a", "doc-b"),
        "member_parent_ids": ("parent-a", "parent-b"),
        "member_chunk_ids": ("chunk-a", "chunk-b"),
        "model_id": "BAAI/bge-m3",
        "model_revision": "revision",
        "tokenizer_id": "BAAI/bge-m3",
        "embedding_config_hash": "embedding-config-hash",
        "dense_vector": (0.1, 0.2, 0.3),
        "dense_dimension": 3,
        "dense_precision": "float32",
        "dense_normalized": True,
        "source_unit_sha256": "source-unit-sha256",
        "source_units_file_sha256": "source-units-file-sha256",
        "adapter_id": "longrag_retrieval_adapter",
        "adapter_version": "0.2.0-source-corrected",
        "source_architecture": "LongRAG",
        "source_architecture_commit": "ba92f95",
        "interface_validation": True,
        "paper_faithful_result": False,
    }
    values.update(overrides)
    return LongUnitEmbeddingRecord(**values)


def test_valid_record_preserves_all_memberships() -> None:
    """Multi-document lineage must remain native."""
    record = make_record()
    assert record.member_doc_ids == ("doc-a", "doc-b")
    assert record.long_unit_id == "longunit-clerk-abc"
    assert record.dense_dimension == 3


def test_dense_dimension_mismatch_is_rejected() -> None:
    """Vector dimensions must be exact."""
    with pytest.raises(ValidationError, match="Dense-vector length"):
        make_record(dense_dimension=4)


def test_duplicate_document_membership_is_rejected() -> None:
    """Document memberships must be unique."""
    with pytest.raises(ValidationError, match="member_doc_ids"):
        make_record(member_doc_ids=("doc-a", "doc-a"))


def test_unsafe_model_input_budget_is_rejected() -> None:
    """Records must prove the text was not truncated."""
    with pytest.raises(ValidationError, match="may truncate"):
        make_record(
            rendered_token_count=4095,
            special_token_count=2,
            actual_model_token_count=4097,
            maximum_model_input_tokens=4096,
        )


def test_model_token_count_mismatch_is_rejected() -> None:
    """Model-token provenance must be internally consistent."""
    with pytest.raises(ValidationError, match="actual_model_token_count"):
        make_record(
            rendered_token_count=4000,
            special_token_count=2,
            actual_model_token_count=4001,
        )


def test_truncated_record_is_rejected() -> None:
    """No truncated embedding can enter the catalog."""
    with pytest.raises(ValidationError, match="truncated"):
        make_record(was_truncated=True)


def test_paper_faithful_claim_is_rejected() -> None:
    """Interface validation must not claim paper faithfulness."""
    with pytest.raises(ValidationError, match="paper-faithful"):
        make_record(paper_faithful_result=True)
