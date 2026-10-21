"""Deterministic synthetic embeddings for infrastructure tests."""

from __future__ import annotations

import hashlib
import math

from rag_system.artifacts import ChildChunk
from rag_system.embedding.models import (
    ChunkEmbeddingRecord,
    SparseVector,
    VectorPrecision,
)


def create_synthetic_embedding(
    *,
    chunk: ChildChunk,
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    embedding_config_hash: str,
    dense_dimension: int,
) -> ChunkEmbeddingRecord:
    """Create deterministic non-semantic test vectors."""

    if dense_dimension < 1:
        raise ValueError("dense_dimension must be positive.")

    digest = hashlib.sha256(chunk.chunk_id.encode("utf-8")).digest()

    raw_values = tuple(
        (digest[index % len(digest)] / 127.5) - 1.0 for index in range(dense_dimension)
    )

    norm = math.sqrt(sum(value * value for value in raw_values))

    if norm == 0:
        raise RuntimeError("Synthetic vector norm cannot be zero.")

    dense_vector = tuple(value / norm for value in raw_values)

    sparse_indices = tuple(
        sorted(
            {
                digest[0],
                digest[1] + 256,
                digest[2] + 512,
            }
        )
    )

    sparse_values = tuple((digest[index] / 255.0) + 0.01 for index in range(len(sparse_indices)))

    return ChunkEmbeddingRecord(
        chunk_id=chunk.chunk_id,
        parent_id=chunk.parent_id,
        doc_id=chunk.doc_id,
        dataset=chunk.dataset,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        embedding_config_hash=(embedding_config_hash),
        dense_vector=dense_vector,
        sparse_vector=SparseVector(
            indices=sparse_indices,
            values=sparse_values,
        ),
        dense_dimension=dense_dimension,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
        source_chunk_sha256=hashlib.sha256(
            chunk.model_dump_json(exclude_none=False).encode("utf-8")
        ).hexdigest(),
    )
