"""Dense-only Qdrant storage helpers for LongRAG long units."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from pydantic import Field, model_validator
from qdrant_client import QdrantClient, models

from rag_system.artifacts import ArtifactBaseModel
from rag_system.baselines.longrag_embeddings import LongUnitEmbeddingRecord
from rag_system.storage import DENSE_VECTOR_NAME

LONGRAG_POINT_ID_PREFIX = "rag-longunit"


class LongRAGCollectionManifest(ArtifactBaseModel):
    """Manifest for one dense-only LongRAG Qdrant collection."""

    artifact_type: str = "longrag_collection_manifest"
    artifact_version: str = "1.0"

    collection_name: str
    storage_mode: str
    storage_path: str

    dataset_id: str
    condition_id: str

    embedding_catalog_path: str
    embedding_catalog_sha256: str
    embedding_config_hash: str

    dense_vector_name: str
    dense_dimension: int = Field(gt=0)
    dense_distance: str

    point_count: int = Field(ge=0)
    expected_point_count: int = Field(gt=0)

    source_unit_count: int = Field(gt=0)
    document_count: int = Field(gt=0)
    shard_count: int = Field(gt=0)

    model_id: str
    model_revision: str
    tokenizer_id: str

    adapter_id: str
    adapter_version: str
    source_architecture: str
    source_architecture_commit: str

    code_revision: str | None = None
    execution_profile: str | None = None

    interface_validation: bool = True
    paper_faithful_result: bool = False

    complete: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> LongRAGCollectionManifest:
        required_values = (
            self.collection_name,
            self.storage_mode,
            self.storage_path,
            self.dataset_id,
            self.condition_id,
            self.embedding_catalog_path,
            self.embedding_catalog_sha256,
            self.embedding_config_hash,
            self.dense_vector_name,
            self.dense_distance,
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.adapter_id,
            self.adapter_version,
            self.source_architecture,
            self.source_architecture_commit,
        )
        if any(not str(v).strip() for v in required_values):
            raise ValueError(
                "LongRAG collection-manifest fields cannot be empty."
            )
        if self.paper_faithful_result:
            raise ValueError(
                "LongRAG collection manifests must not claim paper faithfulness."
            )
        return self


def build_longrag_point_id(
    condition_id: str,
    long_unit_id: str,
    embedding_config_hash: str,
) -> str:
    """Build a deterministic UUID for one long unit in one condition/config."""
    for value in (condition_id, long_unit_id, embedding_config_hash):
        if not value.strip():
            raise ValueError("Point-id ingredients cannot be empty.")
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"{LONGRAG_POINT_ID_PREFIX}:{condition_id}:"
                f"{long_unit_id}:{embedding_config_hash}"
            ),
        )
    )


def build_longrag_payload(record: LongUnitEmbeddingRecord) -> dict[str, Any]:
    """Serialize one embedding record into a JSON-safe Qdrant payload."""
    return {
        "long_unit_id": record.long_unit_id,
        "condition_id": record.condition_id,
        "dataset_id": record.dataset_id,
        "construction_policy": record.construction_policy,
        "target_token_size": record.target_token_size,
        "maximum_model_input_tokens": record.maximum_model_input_tokens,
        "special_token_count": record.special_token_count,
        "actual_model_token_count": record.actual_model_token_count,
        "unit_order": record.unit_order,
        "rendered_token_count": record.rendered_token_count,
        "member_doc_ids": list(record.member_doc_ids),
        "member_parent_ids": list(record.member_parent_ids),
        "member_chunk_ids": list(record.member_chunk_ids),
        "embedding_config_hash": record.embedding_config_hash,
        "adapter_id": record.adapter_id,
        "adapter_version": record.adapter_version,
        "source_architecture": record.source_architecture,
        "source_architecture_commit": record.source_architecture_commit,
        "source_units_file_sha256": record.source_units_file_sha256,
        "interface_validation": record.interface_validation,
        "paper_faithful_result": record.paper_faithful_result,
    }


def build_longrag_point(
    record: LongUnitEmbeddingRecord,
) -> models.PointStruct:
    """Convert one LongUnitEmbeddingRecord to a dense-only Qdrant point."""
    point_id = build_longrag_point_id(
        condition_id=record.condition_id,
        long_unit_id=record.long_unit_id,
        embedding_config_hash=record.embedding_config_hash,
    )
    if len(record.dense_vector) != record.dense_dimension:
        raise ValueError(
            "Dense-vector length does not match dense_dimension."
        )
    return models.PointStruct(
        id=point_id,
        vector={DENSE_VECTOR_NAME: list(record.dense_vector)},
        payload=build_longrag_payload(record),
    )


def create_longrag_collection(
    *,
    client: QdrantClient,
    collection_name: str,
    dense_dimension: int,
    recreate: bool = False,
) -> bool:
    """Create a dense-only cosine collection for LongRAG long units."""
    if dense_dimension <= 0:
        raise ValueError("dense_dimension must be positive.")

    vectors_config = {
        DENSE_VECTOR_NAME: models.VectorParams(
            size=dense_dimension,
            distance=models.Distance.COSINE,
        ),
    }

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if client.collection_exists(collection_name):
        return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    return True


def ingest_longrag_records(
    *,
    client: QdrantClient,
    collection_name: str,
    records: list[LongUnitEmbeddingRecord],
    batch_size: int = 64,
) -> int:
    """Upsert LongRAG embedding records in idempotent batches."""
    if not records:
        raise ValueError("No embedding records supplied.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    seen_ids: set[str] = set()
    points: list[models.PointStruct] = []
    total = 0

    def flush() -> None:
        nonlocal points, total
        if not points:
            return
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        total += len(points)
        points = []

    for record in records:
        point = build_longrag_point(record)
        point_id = str(point.id)
        if point_id in seen_ids:
            raise ValueError(f"Duplicate point id in batch: {point_id}")
        seen_ids.add(point_id)
        points.append(point)
        if len(points) >= batch_size:
            flush()
    flush()
    return total


def fingerprint_longrag_collection_name(
    *,
    dataset_id: str,
    condition_id: str,
    embedding_config_hash: str,
) -> str:
    """Build a deterministic dense-only LongRAG collection name."""
    suffix = hashlib.sha256(
        embedding_config_hash.encode("utf-8")
    ).hexdigest()[:12]
    safe_condition = condition_id.lower().replace("_", "-")
    return f"{dataset_id}_longrag_{safe_condition}_{suffix}"
