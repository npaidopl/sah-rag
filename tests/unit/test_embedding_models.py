"""Tests for embedding artifact models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sah_rag.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingConfig,
    EmbeddingKind,
    EmbeddingShardHeader,
    SparseVector,
    VectorPrecision,
    build_embedding_shard_id,
)


def test_sparse_vector_accepts_sorted_values() -> None:
    """A valid aligned sparse vector should validate."""

    vector = SparseVector(
        indices=(1, 5, 12),
        values=(0.4, 1.2, 0.7),
    )

    assert vector.indices == (1, 5, 12)


def test_sparse_vector_rejects_unsorted_indices() -> None:
    """Sparse indices must use stable sorted order."""

    with pytest.raises(ValidationError):
        SparseVector(
            indices=(5, 1),
            values=(0.4, 0.7),
        )


def test_sparse_vector_rejects_mismatched_arrays() -> None:
    """Sparse indices and values must align."""

    with pytest.raises(ValidationError):
        SparseVector(
            indices=(1, 2),
            values=(0.5,),
        )


def test_hybrid_record_is_valid() -> None:
    """A record may contain dense and sparse vectors."""

    record = ChunkEmbeddingRecord(
        chunk_id="chunk-001",
        parent_id="parent-001",
        doc_id="doc-001",
        dataset="synthetic",
        model_id="BAAI/bge-m3",
        model_revision="revision-001",
        tokenizer_id="tokenizer-001",
        embedding_config_hash="embedding-hash",
        dense_vector=(0.1, 0.2, 0.3),
        sparse_vector=SparseVector(
            indices=(4, 9),
            values=(0.8, 0.2),
        ),
        dense_dimension=3,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
    )

    assert record.dense_dimension == 3
    assert record.sparse_vector is not None


def test_dense_dimension_must_match_vector() -> None:
    """Dense metadata must match the stored vector."""

    with pytest.raises(ValidationError):
        ChunkEmbeddingRecord(
            chunk_id="chunk-001",
            parent_id="parent-001",
            doc_id="doc-001",
            dataset="synthetic",
            model_id="BAAI/bge-m3",
            model_revision="revision-001",
            tokenizer_id="tokenizer-001",
            embedding_config_hash="embedding-hash",
            dense_vector=(0.1, 0.2),
            dense_dimension=3,
            dense_precision=VectorPrecision.FLOAT32,
            dense_normalized=True,
        )


def test_record_requires_one_representation() -> None:
    """An embedding record cannot contain no embeddings."""

    with pytest.raises(ValidationError):
        ChunkEmbeddingRecord(
            chunk_id="chunk-001",
            parent_id="parent-001",
            doc_id="doc-001",
            dataset="synthetic",
            model_id="BAAI/bge-m3",
            model_revision="revision-001",
            tokenizer_id="tokenizer-001",
            embedding_config_hash="embedding-hash",
        )


def test_hybrid_shard_header() -> None:
    """Hybrid shards require dense metadata."""

    header = EmbeddingShardHeader(
        shard_id="embedding-synthetic-000000-abc123",
        shard_index=0,
        dataset="synthetic",
        model_id="BAAI/bge-m3",
        model_revision="revision-001",
        tokenizer_id="tokenizer-001",
        embedding_config_hash="embedding-hash",
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=1024,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
        maximum_records=10000,
    )

    assert header.embedding_kind is EmbeddingKind.HYBRID


def test_embedding_config_rejects_cpu_fp16() -> None:
    """The initial CPU profile should not enable FP16."""

    with pytest.raises(ValidationError):
        EmbeddingConfig(
            model_revision="revision-001",
            tokenizer_id="tokenizer-001",
            device="cpu",
            use_fp16=True,
        )


def test_embedding_shard_id_is_deterministic() -> None:
    """Equivalent shard inputs should produce equal IDs."""

    first = build_embedding_shard_id(
        dataset="clerk",
        model_id="BAAI/bge-m3",
        embedding_config_hash="embedding-hash",
        shard_index=0,
    )

    second = build_embedding_shard_id(
        dataset="clerk",
        model_id="BAAI/bge-m3",
        embedding_config_hash="embedding-hash",
        shard_index=0,
    )

    assert first == second
    assert first.startswith("embedding-clerk-000000-")
