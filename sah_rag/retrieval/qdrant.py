"""Dense, sparse, and RRF retrieval from Qdrant."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from rag_system.retrieval.aggregation import (
    aggregate_chunks_to_parents,
)
from rag_system.retrieval.models import (
    RetrievedChunk,
    RetrievedParent,
)
from rag_system.storage import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)


def _convert_points(
    points: Sequence[models.ScoredPoint],
    *,
    retrieval_method: str,
) -> tuple[RetrievedChunk, ...]:
    """Convert Qdrant points into typed retrieval results."""
    results: list[RetrievedChunk] = []

    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}

        required_fields = (
            "chunk_id",
            "parent_id",
            "doc_id",
            "dataset",
            "text_normalized",
        )

        missing = [
            field
            for field in required_fields
            if not isinstance(payload.get(field), str)
        ]

        if missing:
            raise ValueError(
                "Qdrant result has invalid payload fields: "
                + ", ".join(missing)
            )

        results.append(
            RetrievedChunk(
                chunk_id=str(payload["chunk_id"]),
                parent_id=str(payload["parent_id"]),
                doc_id=str(payload["doc_id"]),
                dataset=str(payload["dataset"]),
                rank=rank,
                score=float(point.score),
                text_normalized=str(
                    payload["text_normalized"]
                ),
                retrieval_method=retrieval_method,
                payload=dict(payload),
            )
        )

    return tuple(results)


def retrieve_dense(
    *,
    client: QdrantClient,
    collection_name: str,
    dense_vector: Sequence[float],
    limit: int,
) -> tuple[RetrievedChunk, ...]:
    """Run named dense-vector retrieval."""
    if limit < 1:
        raise ValueError(
            "Dense retrieval limit must be positive."
        )

    response = client.query_points(
        collection_name=collection_name,
        query=list(dense_vector),
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )

    return _convert_points(
        response.points,
        retrieval_method="dense",
    )


def retrieve_sparse(
    *,
    client: QdrantClient,
    collection_name: str,
    sparse_indices: Sequence[int],
    sparse_values: Sequence[float],
    limit: int,
) -> tuple[RetrievedChunk, ...]:
    """Run named learned-sparse retrieval."""
    if limit < 1:
        raise ValueError(
            "Sparse retrieval limit must be positive."
        )

    if len(sparse_indices) != len(sparse_values):
        raise ValueError(
            "Sparse query indices and values must align."
        )

    response = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(
            indices=list(sparse_indices),
            values=list(sparse_values),
        ),
        using=SPARSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )

    return _convert_points(
        response.points,
        retrieval_method="sparse",
    )


def retrieve_hybrid_rrf(
    *,
    client: QdrantClient,
    collection_name: str,
    dense_vector: Sequence[float],
    sparse_indices: Sequence[int],
    sparse_values: Sequence[float],
    candidate_limit: int,
    result_limit: int,
) -> tuple[RetrievedChunk, ...]:
    """Fuse dense and sparse candidate rankings with RRF."""
    if candidate_limit < result_limit:
        raise ValueError(
            "candidate_limit must be at least result_limit."
        )

    if result_limit < 1:
        raise ValueError(
            "Hybrid result limit must be positive."
        )

    if len(sparse_indices) != len(sparse_values):
        raise ValueError(
            "Sparse query indices and values must align."
        )

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=list(dense_vector),
                using=DENSE_VECTOR_NAME,
                limit=candidate_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(sparse_indices),
                    values=list(sparse_values),
                ),
                using=SPARSE_VECTOR_NAME,
                limit=candidate_limit,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=result_limit,
        with_payload=True,
    )

    return _convert_points(
        response.points,
        retrieval_method="hybrid_rrf",
    )


def retrieve_hybrid_parents(
    *,
    client: QdrantClient,
    collection_name: str,
    dense_vector: Sequence[float],
    sparse_indices: Sequence[int],
    sparse_values: Sequence[float],
    candidate_limit: int,
    child_limit: int,
    parent_limit: int,
    additional_child_weight: float = 0.05,
) -> tuple[RetrievedParent, ...]:
    """Retrieve RRF children and aggregate them to parents."""
    children = retrieve_hybrid_rrf(
        client=client,
        collection_name=collection_name,
        dense_vector=dense_vector,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        candidate_limit=candidate_limit,
        result_limit=child_limit,
    )

    return aggregate_chunks_to_parents(
        children,
        additional_child_weight=(
            additional_child_weight
        ),
        limit=parent_limit,
    )
