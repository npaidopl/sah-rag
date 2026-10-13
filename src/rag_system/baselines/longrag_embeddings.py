"""Typed embedding artifacts for LongRAG-derived retrieval units.

These models preserve native multi-document membership and must not
be replaced with ChunkEmbeddingRecord, whose scalar document identity
is not semantically compatible with adjacent-document long units.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from rag_system.artifacts import ArtifactBaseModel
from rag_system.embedding import (
    EmbeddingShardCatalogEntry,
)


class LongUnitEmbeddingRecord(ArtifactBaseModel):
    """One dense embedding for one LongRAG-derived long unit."""

    artifact_type: str = "longrag_long_unit_embedding"
    artifact_version: str = "1.0"

    long_unit_id: str
    condition_id: str

    dataset_id: str
    construction_policy: str

    target_token_size: int = Field(gt=0)
    maximum_model_input_tokens: int = Field(gt=0)

    unit_order: int = Field(ge=0)
    rendered_token_count: int = Field(gt=0)
    special_token_count: int = Field(ge=0)
    actual_model_token_count: int = Field(gt=0)
    was_truncated: bool = False

    member_doc_ids: tuple[str, ...]
    member_parent_ids: tuple[str, ...]
    member_chunk_ids: tuple[str, ...]

    model_id: str
    model_revision: str
    tokenizer_id: str

    embedding_config_hash: str

    dense_vector: tuple[float, ...]
    dense_dimension: int = Field(gt=0)
    dense_precision: str = "float32"
    dense_normalized: bool = True

    source_unit_sha256: str
    source_units_file_sha256: str

    adapter_id: str
    adapter_version: str

    source_architecture: str = "LongRAG"
    source_architecture_commit: str

    interface_validation: bool = True
    paper_faithful_result: bool = False

    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_embedding(self) -> LongUnitEmbeddingRecord:
        """Validate embedding identity, lineage, and dimensions."""
        required_strings = (
            self.long_unit_id,
            self.condition_id,
            self.dataset_id,
            self.construction_policy,
            self.model_id,
            self.model_revision,
            self.tokenizer_id,
            self.embedding_config_hash,
            self.source_unit_sha256,
            self.source_units_file_sha256,
            self.adapter_id,
            self.adapter_version,
            self.source_architecture,
            self.source_architecture_commit,
        )

        if any(not value.strip() for value in required_strings):
            raise ValueError(
                "Long-unit embedding identity cannot contain empty values."
            )

        if not self.member_doc_ids:
            raise ValueError("member_doc_ids cannot be empty.")
        if not self.member_parent_ids:
            raise ValueError("member_parent_ids cannot be empty.")
        if not self.member_chunk_ids:
            raise ValueError("member_chunk_ids cannot be empty.")

        if len(self.member_doc_ids) != len(set(self.member_doc_ids)):
            raise ValueError("member_doc_ids must be unique.")
        if len(self.member_parent_ids) != len(set(self.member_parent_ids)):
            raise ValueError("member_parent_ids must be unique.")
        if len(self.member_chunk_ids) != len(set(self.member_chunk_ids)):
            raise ValueError("member_chunk_ids must be unique.")

        if len(self.dense_vector) != self.dense_dimension:
            raise ValueError(
                "Dense-vector length does not match dense_dimension."
            )

        if self.rendered_token_count > self.target_token_size:
            raise ValueError(
                "Rendered token count exceeds the condition target."
            )

        expected_model_token_count = (
            self.rendered_token_count + self.special_token_count
        )

        if self.actual_model_token_count != expected_model_token_count:
            raise ValueError(
                "actual_model_token_count does not match "
                "rendered tokens plus special tokens."
            )

        if self.actual_model_token_count > self.maximum_model_input_tokens:
            raise ValueError(
                "Model input budget may truncate the rendered unit."
            )

        if self.was_truncated:
            raise ValueError(
                "Long-unit embedding records cannot represent truncated inputs."
            )

        if self.paper_faithful_result:
            raise ValueError(
                "The current adapter must not mark artifacts as paper-faithful."
            )

        return self


class LongUnitEmbeddingShardHeader(ArtifactBaseModel):
    """Header for one long-unit embedding shard."""

    artifact_type: str = "longrag_embedding_shard_header"
    artifact_version: str = "1.0"

    shard_id: str
    shard_index: int = Field(ge=0)

    dataset_id: str
    condition_id: str

    model_id: str
    model_revision: str
    tokenizer_id: str

    embedding_config_hash: str

    dense_dimension: int = Field(gt=0)
    dense_precision: str
    dense_normalized: bool

    maximum_model_input_tokens: int = Field(gt=0)
    special_token_count: int = Field(ge=0)
    maximum_records: int = Field(gt=0)

    adapter_id: str
    adapter_version: str

    source_architecture_commit: str
    source_units_path: str
    source_units_sha256: str

    interface_validation: bool = True
    paper_faithful_result: bool = False


class LongUnitEmbeddingCatalog(ArtifactBaseModel):
    """Catalog of one condition's embedding shards."""

    artifact_type: str = "longrag_embedding_run_catalog"
    artifact_version: str = "1.0"

    run_id: str

    dataset_id: str
    condition_id: str

    construction_policy: str
    target_token_size: int = Field(gt=0)
    maximum_model_input_tokens: int = Field(gt=0)
    special_token_count: int = Field(ge=0)

    model_id: str
    model_revision: str
    tokenizer_id: str

    embedding_config_hash: str

    dense_dimension: int = Field(gt=0)
    dense_precision: str
    dense_normalized: bool

    source_units_path: str
    source_units_sha256: str

    shard_count: int = Field(gt=0)
    record_count: int = Field(gt=0)
    document_count: int = Field(gt=0)

    shards: tuple[EmbeddingShardCatalogEntry, ...]

    adapter_id: str
    adapter_version: str

    source_architecture: str
    source_architecture_commit: str

    code_revision: str | None = None
    execution_profile: str | None = None

    interface_validation: bool = True
    paper_faithful_result: bool = False

    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_catalog(self) -> LongUnitEmbeddingCatalog:
        """Validate aggregate shard metadata."""
        if self.shard_count != len(self.shards):
            raise ValueError("shard_count does not match shards.")

        if sum(shard.record_count for shard in self.shards) != self.record_count:
            raise ValueError("Shard records do not match catalog.")

        if self.paper_faithful_result:
            raise ValueError(
                "The current adapter must not mark the catalog as paper-faithful."
            )

        return self
