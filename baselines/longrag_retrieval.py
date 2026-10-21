"""Dense retrieval over LongRAG long-unit collections."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from rag_system.artifacts import ArtifactBaseModel
from rag_system.baselines.longrag import (
    LongUnitConstructionPolicy,
    RankedLongUnit,
)
from rag_system.storage import DENSE_VECTOR_NAME


class RetrievedLongUnit(ArtifactBaseModel):
    """One dense-retrieved long unit with payload and rank."""

    long_unit_id: str
    condition_id: str
    dataset_id: str

    rank: int
    score: float

    construction_policy: LongUnitConstructionPolicy
    target_token_size: int
    rendered_token_count: int

    member_doc_ids: tuple[str, ...]
    member_parent_ids: tuple[str, ...]
    member_chunk_ids: tuple[str, ...]

    payload: dict[str, Any]


def _coerce_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(x) for x in value)
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    return ()


def _point_to_retrieved(
    point: models.ScoredPoint,
    rank: int,
) -> RetrievedLongUnit:
    payload = dict(point.payload or {})
    required = (
        "long_unit_id",
        "condition_id",
        "dataset_id",
        "construction_policy",
        "target_token_size",
        "rendered_token_count",
        "member_doc_ids",
        "member_parent_ids",
        "member_chunk_ids",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"Qdrant payload is missing required fields: {missing}"
        )
    return RetrievedLongUnit(
        long_unit_id=str(payload["long_unit_id"]),
        condition_id=str(payload["condition_id"]),
        dataset_id=str(payload["dataset_id"]),
        rank=rank,
        score=float(point.score),
        construction_policy=payload["construction_policy"],
        target_token_size=int(payload["target_token_size"]),
        rendered_token_count=int(payload["rendered_token_count"]),
        member_doc_ids=_coerce_tuple(payload["member_doc_ids"]),
        member_parent_ids=_coerce_tuple(payload["member_parent_ids"]),
        member_chunk_ids=_coerce_tuple(payload["member_chunk_ids"]),
        payload=payload,
    )


def retrieve_long_units(
    *,
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float] | tuple[float, ...],
    limit: int = 10,
    condition_id: str | None = None,
) -> tuple[RetrievedLongUnit, ...]:
    """Query a dense-only LongRAG collection and return typed results."""
    if limit <= 0:
        raise ValueError("Retrieval limit must be positive.")

    query_filter: models.Filter | None = None
    if condition_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="condition_id",
                    match=models.MatchValue(value=condition_id),
                )
            ]
        )

    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )

    return tuple(
        _point_to_retrieved(point, rank)
        for rank, point in enumerate(response.points, start=1)
    )


def retrieved_to_ranked_long_units(
    *,
    query_id: str,
    retrieved: tuple[RetrievedLongUnit, ...],
) -> tuple[RankedLongUnit, ...]:
    """Convert retrieved units into the adapter RankedLongUnit contract."""
    return tuple(
        RankedLongUnit(
            query_id=query_id,
            rank=item.rank,
            long_unit_id=item.long_unit_id,
            score=item.score,
            construction_policy=item.construction_policy,
            member_doc_ids=item.member_doc_ids,
            member_parent_ids=item.member_parent_ids,
            member_chunk_ids=item.member_chunk_ids,
        )
        for item in retrieved
    )
