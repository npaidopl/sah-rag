"""Tests for deterministic Qdrant storage utilities."""

from __future__ import annotations

from qdrant_client import QdrantClient

from rag_system.artifacts import ChildChunk, OffsetSpan
from rag_system.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingKind,
    EmbeddingShardHeader,
    SparseVector,
    VectorPrecision,
)
from rag_system.storage import (
    build_point,
    build_qdrant_point_id,
    create_hybrid_collection,
    fingerprint_collection_name,
    ingest_embedding_records,
)


def make_chunk(
    chunk_id: str = "chunk-001",
) -> ChildChunk:
    """Create one source chunk."""
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id="parent-001",
        doc_id="doc-001",
        dataset="synthetic",
        chunk_index=0,
        parent_chunk_index=0,
        text_normalized="Example source chunk.",
        text_raw="Example source chunk.",
        normalized_char_span=OffsetSpan(
            start=0,
            end=21,
        ),
        parent_normalized_char_span=OffsetSpan(
            start=0,
            end=21,
        ),
        raw_char_span=OffsetSpan(
            start=0,
            end=21,
        ),
        token_span=OffsetSpan(
            start=0,
            end=4,
        ),
        token_count=4,
        parser_id="parser",
        parser_version="0.3.0",
        parser_config_hash="parser-hash",
        chunker_id="chunker",
        chunker_version="0.3.0",
        chunker_config_hash="chunker-hash",
        tokenizer_id="tokenizer",
        chunk_size=256,
        chunk_overlap=32,
        metadata={},
    )


def make_header() -> EmbeddingShardHeader:
    """Create one hybrid shard header."""
    return EmbeddingShardHeader(
        shard_id="embedding-synthetic-000000-test",
        shard_index=0,
        dataset="synthetic",
        model_id="test/model",
        model_revision="revision-001",
        tokenizer_id="tokenizer",
        embedding_config_hash="embedding-hash",
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=3,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
        maximum_records=10,
    )


def make_record(
    chunk_id: str = "chunk-001",
) -> ChunkEmbeddingRecord:
    """Create one hybrid embedding record."""
    return ChunkEmbeddingRecord(
        chunk_id=chunk_id,
        parent_id="parent-001",
        doc_id="doc-001",
        dataset="synthetic",
        model_id="test/model",
        model_revision="revision-001",
        tokenizer_id="tokenizer",
        embedding_config_hash="embedding-hash",
        dense_vector=(1.0, 0.0, 0.0),
        sparse_vector=SparseVector(
            indices=(1, 7),
            values=(0.8, 0.2),
        ),
        dense_dimension=3,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
    )


def test_qdrant_point_id_is_deterministic() -> None:
    """Equivalent chunk IDs must produce equal UUIDs."""
    first = build_qdrant_point_id("chunk-001")
    second = build_qdrant_point_id("chunk-001")
    different = build_qdrant_point_id("chunk-002")

    assert first == second
    assert first != different


def test_collection_name_is_deterministic() -> None:
    """Collection identity must depend on dataset and config."""
    first = fingerprint_collection_name(
        dataset="Greek Bar Bench",
        embedding_config_hash="hash-001",
    )
    second = fingerprint_collection_name(
        dataset="Greek Bar Bench",
        embedding_config_hash="hash-001",
    )

    assert first == second
    assert first.startswith("rag_greek_bar_bench_")


def test_build_point_preserves_payload() -> None:
    """A Qdrant point must preserve source identities."""
    point = build_point(
        chunk=make_chunk(),
        record=make_record(),
        header=make_header(),
    )

    assert point.payload is not None
    assert point.payload["chunk_id"] == "chunk-001"
    assert point.payload["parent_id"] == "parent-001"
    assert point.payload["doc_id"] == "doc-001"
    assert point.payload["dataset"] == "synthetic"


def test_in_memory_collection_ingestion() -> None:
    """Hybrid vectors should be inserted into local Qdrant."""
    client = QdrantClient(":memory:")
    collection_name = "test_hybrid_collection"

    create_hybrid_collection(
        client=client,
        collection_name=collection_name,
        dense_dimension=3,
    )

    summary = ingest_embedding_records(
        client=client,
        collection_name=collection_name,
        header=make_header(),
        records=[make_record()],
        chunks=[make_chunk()],
        batch_size=1,
    )

    count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    assert summary.point_count == 1
    assert count == 1
    assert summary.dense_dimension == 3
