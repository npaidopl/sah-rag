"""Integration tests for Qdrant RRF and parent retrieval."""

from __future__ import annotations

from qdrant_client import QdrantClient

from sah_rag.artifacts import ChildChunk, OffsetSpan
from sah_rag.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingKind,
    EmbeddingShardHeader,
    SparseVector,
    VectorPrecision,
)
from sah_rag.retrieval import (
    retrieve_hybrid_parents,
    retrieve_hybrid_rrf,
)
from sah_rag.storage import (
    create_hybrid_collection,
    ingest_embedding_records,
)


def make_chunk(
    *,
    chunk_id: str,
    parent_id: str,
    chunk_index: int,
) -> ChildChunk:
    """Create one source chunk."""
    text = f"Source text for {chunk_id}."

    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        doc_id=f"doc-{parent_id}",
        dataset="synthetic",
        chunk_index=chunk_index,
        parent_chunk_index=chunk_index,
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


def make_record(
    *,
    chunk: ChildChunk,
    dense_vector: tuple[float, ...],
    sparse_index: int,
    sparse_value: float,
) -> ChunkEmbeddingRecord:
    """Create one hybrid embedding record."""
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
            values=(sparse_value,),
        ),
        dense_dimension=3,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
    )


def build_test_collection() -> tuple[
    QdrantClient,
    str,
    list[ChildChunk],
]:
    """Create one populated in-memory hybrid collection."""
    chunks = [
        make_chunk(
            chunk_id="chunk-a1",
            parent_id="parent-a",
            chunk_index=0,
        ),
        make_chunk(
            chunk_id="chunk-a2",
            parent_id="parent-a",
            chunk_index=1,
        ),
        make_chunk(
            chunk_id="chunk-b1",
            parent_id="parent-b",
            chunk_index=2,
        ),
    ]

    records = [
        make_record(
            chunk=chunks[0],
            dense_vector=(1.0, 0.0, 0.0),
            sparse_index=10,
            sparse_value=1.0,
        ),
        make_record(
            chunk=chunks[1],
            dense_vector=(0.9, 0.1, 0.0),
            sparse_index=10,
            sparse_value=0.8,
        ),
        make_record(
            chunk=chunks[2],
            dense_vector=(0.0, 1.0, 0.0),
            sparse_index=20,
            sparse_value=1.0,
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
    collection_name = "rrf_integration"

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

    return client, collection_name, chunks


def test_rrf_ranks_joint_dense_sparse_match_first() -> None:
    """RRF should prioritize a result strong in both rankings."""
    client, collection_name, _ = build_test_collection()

    results = retrieve_hybrid_rrf(
        client=client,
        collection_name=collection_name,
        dense_vector=(1.0, 0.0, 0.0),
        sparse_indices=(10,),
        sparse_values=(1.0,),
        candidate_limit=3,
        result_limit=3,
    )

    assert len(results) == 3
    assert results[0].chunk_id == "chunk-a1"
    assert results[0].retrieval_method == "hybrid_rrf"

    assert [
        result.rank
        for result in results
    ] == [1, 2, 3]


def test_hybrid_parent_aggregation_has_unique_parents() -> None:
    """Parent aggregation should merge sibling chunks."""
    client, collection_name, _ = build_test_collection()

    parents = retrieve_hybrid_parents(
        client=client,
        collection_name=collection_name,
        dense_vector=(1.0, 0.0, 0.0),
        sparse_indices=(10,),
        sparse_values=(1.0,),
        candidate_limit=3,
        child_limit=3,
        parent_limit=2,
    )

    assert len(parents) == 2

    assert len({
        parent.parent_id
        for parent in parents
    }) == len(parents)

    assert parents[0].parent_id == "parent-a"
    assert parents[0].matched_chunk_count == 2
    assert set(parents[0].chunk_ids) == {
        "chunk-a1",
        "chunk-a2",
    }
