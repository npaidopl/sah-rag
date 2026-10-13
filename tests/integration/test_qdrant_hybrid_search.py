"""Integration test for dense and sparse Qdrant search."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from rag_system.artifacts import ChildChunk, OffsetSpan
from rag_system.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingKind,
    EmbeddingShardHeader,
    SparseVector,
    VectorPrecision,
)
from rag_system.storage import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    create_hybrid_collection,
    ingest_embedding_records,
)


def make_chunk(
    *,
    chunk_id: str,
    chunk_index: int,
    text: str,
) -> ChildChunk:
    """Create one source chunk."""
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_index}",
        doc_id=f"doc-{chunk_index}",
        dataset="synthetic",
        chunk_index=chunk_index,
        parent_chunk_index=0,
        text_normalized=text,
        text_raw=text,
        normalized_char_span=OffsetSpan(
            start=0,
            end=len(text),
        ),
        parent_normalized_char_span=OffsetSpan(
            start=0,
            end=len(text),
        ),
        raw_char_span=OffsetSpan(
            start=0,
            end=len(text),
        ),
        token_span=OffsetSpan(
            start=0,
            end=2,
        ),
        token_count=2,
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


def make_record(
    *,
    chunk: ChildChunk,
    dense_vector: tuple[float, ...],
    sparse_index: int,
) -> ChunkEmbeddingRecord:
    """Create one hybrid embedding."""
    return ChunkEmbeddingRecord(
        chunk_id=chunk.chunk_id,
        parent_id=chunk.parent_id,
        doc_id=chunk.doc_id,
        dataset=chunk.dataset,
        model_id="test/model",
        model_revision="revision-001",
        tokenizer_id="tokenizer",
        embedding_config_hash="embedding-hash",
        dense_vector=dense_vector,
        sparse_vector=SparseVector(
            indices=(sparse_index,),
            values=(1.0,),
        ),
        dense_dimension=3,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
    )


def test_dense_and_sparse_self_search() -> None:
    """Each representation should retrieve itself first."""
    chunks = [
        make_chunk(
            chunk_id="chunk-a",
            chunk_index=0,
            text="Alpha text.",
        ),
        make_chunk(
            chunk_id="chunk-b",
            chunk_index=1,
            text="Beta text.",
        ),
    ]

    records = [
        make_record(
            chunk=chunks[0],
            dense_vector=(1.0, 0.0, 0.0),
            sparse_index=10,
        ),
        make_record(
            chunk=chunks[1],
            dense_vector=(0.0, 1.0, 0.0),
            sparse_index=20,
        ),
    ]

    header = EmbeddingShardHeader(
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

    client = QdrantClient(":memory:")
    collection_name = "integration_hybrid"

    create_hybrid_collection(
        client=client,
        collection_name=collection_name,
        dense_dimension=3,
    )

    ingest_embedding_records(
        client=client,
        collection_name=collection_name,
        header=header,
        records=records,
        chunks=chunks,
    )

    dense_response = client.query_points(
        collection_name=collection_name,
        query=[1.0, 0.0, 0.0],
        using=DENSE_VECTOR_NAME,
        limit=1,
        with_payload=True,
    )

    sparse_response = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(
            indices=[10],
            values=[1.0],
        ),
        using=SPARSE_VECTOR_NAME,
        limit=1,
        with_payload=True,
    )

    dense_payload = dense_response.points[0].payload or {}
    sparse_payload = sparse_response.points[0].payload or {}

    assert dense_payload["chunk_id"] == "chunk-a"
    assert sparse_payload["chunk_id"] == "chunk-a"
