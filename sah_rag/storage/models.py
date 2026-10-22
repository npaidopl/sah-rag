"""Typed artifacts describing persistent vector collections."""

from __future__ import annotations

from pydantic import Field, model_validator

from sah_rag.artifacts import ArtifactBaseModel


class QdrantCollectionManifest(ArtifactBaseModel):
    """Identity and integrity metadata for one Qdrant collection."""

    artifact_type: str = "qdrant_collection_manifest"
    artifact_version: str = "1.0"

    collection_name: str
    storage_mode: str
    storage_path: str

    dataset_id: str

    embedding_catalog_path: str
    embedding_catalog_sha256: str
    embedding_config_hash: str

    dense_vector_name: str
    dense_dimension: int = Field(gt=0)
    dense_distance: str

    sparse_vector_name: str
    sparse_distance: str

    point_count: int = Field(ge=0)
    expected_point_count: int = Field(gt=0)

    source_chunk_count: int = Field(gt=0)
    document_count: int = Field(gt=0)
    shard_count: int = Field(gt=0)

    model_id: str
    model_revision: str
    tokenizer_id: str

    code_revision: str | None = None
    execution_profile: str | None = None

    complete: bool = False

    metadata: dict[str, object] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_manifest(
        self,
    ) -> QdrantCollectionManifest:
        """Validate collection identity and completeness."""

        required_values = (
            self.collection_name,
            self.storage_mode,
            self.storage_path,
            self.dataset_id,
            self.embedding_catalog_path,
            self.embedding_config_hash,
            self.dense_vector_name,
            self.dense_distance,
            self.sparse_vector_name,
            self.sparse_distance,
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
        )

        if any(
            not value.strip()
            for value in required_values
        ):
            raise ValueError(
                "Collection-manifest fields cannot be empty."
            )

        for fingerprint in (
            self.embedding_catalog_sha256,
            self.embedding_config_hash,
        ):
            if len(fingerprint) != 64:
                raise ValueError(
                    "Collection fingerprints must be SHA-256."
                )

            try:
                int(fingerprint, 16)
            except ValueError as exc:
                raise ValueError(
                    "Collection fingerprint is not hexadecimal."
                ) from exc

        if self.complete and (
            self.point_count
            != self.expected_point_count
        ):
            raise ValueError(
                "A complete collection must contain the "
                "expected number of points."
            )

        return self
