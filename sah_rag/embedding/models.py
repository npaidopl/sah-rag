"""Typed artifacts for dense and learned-sparse embeddings."""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field, model_validator

from sah_rag.artifacts.models import ArtifactBaseModel


class EmbeddingKind(StrEnum):
    """Embedding representations supported by the pipeline."""

    DENSE = "dense"
    LEARNED_SPARSE = "learned_sparse"
    HYBRID = "hybrid"


class VectorPrecision(StrEnum):
    """Supported dense-vector storage precision."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"


class SparseVector(ArtifactBaseModel):
    """One sparse vector represented by aligned arrays."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def validate_sparse_vector(self) -> SparseVector:
        """Validate sparse-vector alignment and values."""

        if len(self.indices) != len(self.values):
            raise ValueError("Sparse indices and values must have the same length.")

        if any(index < 0 for index in self.indices):
            raise ValueError("Sparse indices cannot be negative.")

        if len(self.indices) != len(set(self.indices)):
            raise ValueError("Sparse indices must be unique.")

        if tuple(sorted(self.indices)) != self.indices:
            raise ValueError("Sparse indices must be sorted.")

        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("Sparse values must be finite.")

        return self


class ChunkEmbeddingRecord(ArtifactBaseModel):
    """Embeddings generated for one child chunk."""

    artifact_type: str = "chunk_embedding"
    artifact_version: str = "1.0"

    chunk_id: str
    parent_id: str
    doc_id: str
    dataset: str

    model_id: str
    model_revision: str
    tokenizer_id: str
    embedding_config_hash: str

    dense_vector: tuple[float, ...] | None = None
    sparse_vector: SparseVector | None = None

    dense_dimension: int | None = Field(
        default=None,
        gt=0,
    )
    dense_precision: VectorPrecision | None = None
    dense_normalized: bool | None = None

    source_chunk_sha256: str | None = None

    @model_validator(mode="after")
    def validate_embedding_record(
        self,
    ) -> ChunkEmbeddingRecord:
        """Validate identities and vector representations."""

        required_values = (
            self.chunk_id,
            self.parent_id,
            self.doc_id,
            self.dataset,
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.embedding_config_hash,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError("Embedding identity fields cannot be empty.")

        if self.dense_vector is None and self.sparse_vector is None:
            raise ValueError("At least one embedding representation is required.")

        if self.dense_vector is not None:
            if self.dense_dimension is None:
                raise ValueError("Dense vectors require dense_dimension.")

            if len(self.dense_vector) != self.dense_dimension:
                raise ValueError("Dense-vector length does not match dense_dimension.")

            if self.dense_precision is None:
                raise ValueError("Dense vectors require dense_precision.")

            if self.dense_normalized is None:
                raise ValueError("Dense vectors require normalization metadata.")

            if any(not math.isfinite(value) for value in self.dense_vector):
                raise ValueError("Dense-vector values must be finite.")
        else:
            if any(
                value is not None
                for value in (
                    self.dense_dimension,
                    self.dense_precision,
                    self.dense_normalized,
                )
            ):
                raise ValueError("Dense metadata cannot exist without a dense vector.")

        if self.source_chunk_sha256 is not None and len(self.source_chunk_sha256) != 64:
            raise ValueError("source_chunk_sha256 must contain 64 hexadecimal characters.")

        if self.source_chunk_sha256 is not None:
            try:
                int(self.source_chunk_sha256, 16)
            except ValueError as exc:
                raise ValueError("source_chunk_sha256 is not hexadecimal.") from exc

        return self


class EmbeddingShardHeader(ArtifactBaseModel):
    """Identity and configuration shared by one embedding shard."""

    artifact_type: str = "embedding_shard_header"
    artifact_version: str = "1.0"

    shard_id: str
    shard_index: int = Field(ge=0)

    dataset: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    embedding_config_hash: str

    embedding_kind: EmbeddingKind

    dense_dimension: int | None = Field(
        default=None,
        gt=0,
    )
    dense_precision: VectorPrecision | None = None
    dense_normalized: bool | None = None

    maximum_records: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_header(
        self,
    ) -> EmbeddingShardHeader:
        """Validate shard identity and representation metadata."""

        required_values = (
            self.shard_id,
            self.dataset,
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.embedding_config_hash,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError("Embedding-shard fields cannot be empty.")

        dense_required = self.embedding_kind in {
            EmbeddingKind.DENSE,
            EmbeddingKind.HYBRID,
        }

        if dense_required and any(
            value is None
            for value in (
                self.dense_dimension,
                self.dense_precision,
                self.dense_normalized,
            )
        ):
            raise ValueError("Dense or hybrid shards require complete dense-vector metadata.")

        if not dense_required and any(
            value is not None
            for value in (
                self.dense_dimension,
                self.dense_precision,
                self.dense_normalized,
            )
        ):
            raise ValueError("Sparse-only shards cannot contain dense-vector metadata.")

        return self
