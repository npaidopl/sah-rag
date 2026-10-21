"""Qdrant collection creation and hybrid-vector ingestion."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from rag_system.artifacts import ChildChunk
from rag_system.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingShardHeader,
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass(frozen=True, slots=True)
class QdrantIngestionSummary:
    """Summary of one embedding-shard ingestion."""

    collection_name: str
    point_count: int
    dense_dimension: int
    dense_vector_name: str
    sparse_vector_name: str


def build_qdrant_point_id(chunk_id: str) -> str:
    """Build a deterministic UUID from a chunk identifier."""
    if not chunk_id.strip():
        raise ValueError("chunk_id cannot be empty.")

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rag-chunk:{chunk_id}",
        )
    )


def create_hybrid_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    dense_dimension: int,
    recreate: bool = False,
) -> None:
    """Create a named dense and sparse vector collection."""
    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty.")

    if dense_dimension < 1:
        raise ValueError("dense_dimension must be positive.")

    exists = client.collection_exists(
        collection_name=collection_name
    )

    if exists and recreate:
        client.delete_collection(
            collection_name=collection_name
        )
        exists = False

    if exists:
        raise ValueError(
            f"Collection already exists: {collection_name}"
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dense_dimension,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: (
                models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            ),
        },
    )


def build_payload(
    *,
    chunk: ChildChunk,
    record: ChunkEmbeddingRecord,
    header: EmbeddingShardHeader,
) -> dict[str, Any]:
    """Build the searchable payload for one Qdrant point."""
    if chunk.chunk_id != record.chunk_id:
        raise ValueError("Chunk and embedding IDs do not match.")

    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "doc_id": chunk.doc_id,
        "dataset": chunk.dataset,
        "chunk_index": chunk.chunk_index,
        "parent_chunk_index": (
            chunk.parent_chunk_index
        ),
        "text_normalized": chunk.text_normalized,
        "normalized_char_start": (
            chunk.normalized_char_span.start
        ),
        "normalized_char_end": (
            chunk.normalized_char_span.end
        ),
        "token_start": (
            chunk.token_span.start
            if chunk.token_span is not None
            else None
        ),
        "token_end": (
            chunk.token_span.end
            if chunk.token_span is not None
            else None
        ),
        "token_count": chunk.token_count,
        "model_id": record.model_id,
        "model_revision": record.model_revision,
        "embedding_config_hash": (
            record.embedding_config_hash
        ),
        "embedding_shard_id": header.shard_id,
    }


def build_point(
    *,
    chunk: ChildChunk,
    record: ChunkEmbeddingRecord,
    header: EmbeddingShardHeader,
) -> models.PointStruct:
    """Convert one embedding record to a Qdrant point."""
    if record.dense_vector is None:
        raise ValueError("Qdrant hybrid ingestion requires a dense vector.")

    if record.sparse_vector is None:
        raise ValueError("Qdrant hybrid ingestion requires a sparse vector.")

    if (
        header.dense_dimension is None
        or len(record.dense_vector)
        != header.dense_dimension
    ):
        raise ValueError("Dense-vector dimension does not match the header.")

    return models.PointStruct(
        id=build_qdrant_point_id(chunk.chunk_id),
        vector={
            DENSE_VECTOR_NAME: list(
                record.dense_vector
            ),
            SPARSE_VECTOR_NAME: models.SparseVector(
                indices=list(
                    record.sparse_vector.indices
                ),
                values=list(
                    record.sparse_vector.values
                ),
            ),
        },
        payload=build_payload(
            chunk=chunk,
            record=record,
            header=header,
        ),
    )


def ingest_embedding_records(
    *,
    client: QdrantClient,
    collection_name: str,
    header: EmbeddingShardHeader,
    records: Iterable[ChunkEmbeddingRecord],
    chunks: Iterable[ChildChunk],
    batch_size: int = 64,
) -> QdrantIngestionSummary:
    """Validate and insert embedding records into Qdrant."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if header.dense_dimension is None:
        raise ValueError("Hybrid ingestion requires dense_dimension.")

    chunk_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    if len(chunk_by_id) == 0:
        raise ValueError("No source chunks were supplied.")

    points: list[models.PointStruct] = []
    point_ids: set[str] = set()
    total_count = 0

    for record in records:
        chunk = chunk_by_id.get(record.chunk_id)

        if chunk is None:
            raise ValueError(
                "Embedding references an unknown source "
                f"chunk: {record.chunk_id}"
            )

        point = build_point(
            chunk=chunk,
            record=record,
            header=header,
        )

        point_id = str(point.id)

        if point_id in point_ids:
            raise ValueError(
                f"Duplicate Qdrant point ID: {point_id}"
            )

        point_ids.add(point_id)
        points.append(point)

        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            total_count += len(points)
            points = []

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        total_count += len(points)

    return QdrantIngestionSummary(
        collection_name=collection_name,
        point_count=total_count,
        dense_dimension=header.dense_dimension,
        dense_vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def fingerprint_collection_name(
    *,
    dataset: str,
    embedding_config_hash: str,
) -> str:
    """Build a deterministic Qdrant collection name."""
    suffix = hashlib.sha256(
        embedding_config_hash.encode("utf-8")
    ).hexdigest()[:12]

    normalized_dataset = "".join(
        character.lower()
        if character.isalnum()
        else "_"
        for character in dataset
    ).strip("_")

    return f"rag_{normalized_dataset}_{suffix}"
