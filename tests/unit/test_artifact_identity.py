"""Tests for deterministic artifact identities."""

from __future__ import annotations

import pytest

from rag_system.artifacts.identity import (
    build_chunk_id,
    build_experiment_id,
    build_parent_id,
    normalize_identifier,
    sha256_fingerprint,
    short_fingerprint,
)


def test_fingerprint_ignores_mapping_order() -> None:
    """Equivalent mappings should have equal fingerprints."""
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    assert sha256_fingerprint(first) == sha256_fingerprint(
        second
    )


def test_short_fingerprint_is_deterministic() -> None:
    """Short fingerprints should remain stable."""
    value = {"chunk_size": 256, "overlap": 32}
    assert short_fingerprint(value) == short_fingerprint(value)
    assert len(short_fingerprint(value)) == 12


def test_short_fingerprint_rejects_invalid_length() -> None:
    """Unsafe fingerprint lengths should fail."""
    with pytest.raises(ValueError):
        short_fingerprint({"a": 1}, length=4)


def test_normalize_identifier() -> None:
    """Human-readable names should become stable IDs."""
    assert (
        normalize_identifier("Greek Bar Bench")
        == "greek-bar-bench"
    )


def test_parent_id_is_deterministic() -> None:
    """Equivalent parent identity inputs should match."""
    first = build_parent_id(
        dataset="cuad",
        doc_id="doc-001",
        parser_config_hash="parser-hash",
        parent_index=3,
    )
    second = build_parent_id(
        dataset="cuad",
        doc_id="doc-001",
        parser_config_hash="parser-hash",
        parent_index=3,
    )
    assert first == second
    assert first.startswith("parent-cuad-000003-")


def test_chunk_id_changes_with_parent_chunk_index() -> None:
    """Different offsets in a parent must create distinct IDs."""
    first = build_chunk_id(
        dataset="cuad",
        doc_id="doc-001",
        parent_id="parent-001",
        chunker_config_hash="chunker-hash",
        parent_chunk_index=0,
    )
    second = build_chunk_id(
        dataset="cuad",
        doc_id="doc-001",
        parent_id="parent-001",
        chunker_config_hash="chunker-hash",
        parent_chunk_index=1,
    )
    assert first != second


def test_experiment_id_is_deterministic() -> None:
    """Equivalent experiment configurations should match."""
    first = build_experiment_id(
        architecture_id="Proposed RAG",
        dataset_id="CUAD",
        configuration={
            "chunk_size": 256,
            "top_k": 10,
        },
    )
    second = build_experiment_id(
        architecture_id="Proposed RAG",
        dataset_id="CUAD",
        configuration={
            "top_k": 10,
            "chunk_size": 256,
        },
    )
    assert first == second
    assert first.startswith("proposed-rag__cuad__")