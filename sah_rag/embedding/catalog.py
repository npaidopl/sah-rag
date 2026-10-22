"""Catalog models and validation for embedding-shard runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from sah_rag.artifacts import (
    ArtifactBaseModel,
    ArtifactSidecarManifest,
    read_json_model,
    sha256_file,
)
from sah_rag.embedding.models import EmbeddingShardHeader
from sah_rag.embedding.shards import iter_embedding_records


class EmbeddingShardCatalogEntry(ArtifactBaseModel):
    """Catalog identity and integrity data for one shard."""
    shard_id: str
    shard_index: int = Field(ge=0)
    shard_root: str
    header_path: str
    records_path: str
    manifest_path: str
    header_sha256: str
    records_sha256: str
    manifest_sha256: str
    record_count: int = Field(gt=0)
    source_start_index: int = Field(ge=0)
    source_end_index: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_entry(self) -> EmbeddingShardCatalogEntry:
        if self.source_end_index <= self.source_start_index:
            raise ValueError("Shard source range must be positive.")
        if self.source_end_index - self.source_start_index != self.record_count:
            raise ValueError("Shard source range does not match record_count.")
        for value in (self.header_sha256, self.records_sha256, self.manifest_sha256):
            if len(value) != 64:
                raise ValueError("Catalog SHA-256 values must contain 64 characters.")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError("Catalog SHA-256 value is not hexadecimal.") from exc
        return self


class EmbeddingRunCatalog(ArtifactBaseModel):
    """Validated catalog for one complete embedding run."""
    artifact_type: str = "embedding_run_catalog"
    artifact_version: str = "1.0"
    run_id: str
    dataset_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    embedding_config_hash: str
    embedding_kind: str
    dense_dimension: int = Field(gt=0)
    dense_precision: str
    dense_normalized: bool
    source_chunks_path: str
    source_chunks_manifest_path: str
    source_chunks_sha256: str
    shard_count: int = Field(gt=0)
    record_count: int = Field(gt=0)
    document_count: int = Field(gt=0)
    shards: tuple[EmbeddingShardCatalogEntry, ...]
    code_revision: str | None = None
    execution_profile: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_catalog(self) -> EmbeddingRunCatalog:
        if len(self.embedding_config_hash) != 64:
            raise ValueError("embedding_config_hash must be SHA-256.")
        if len(self.shards) != self.shard_count:
            raise ValueError("shard_count does not match catalog entries.")
        indices = tuple(entry.shard_index for entry in self.shards)
        if indices != tuple(range(self.shard_count)):
            raise ValueError("Shard indices must be contiguous from zero.")
        if sum(entry.record_count for entry in self.shards) != self.record_count:
            raise ValueError("Catalog shard records do not match record_count.")
        expected_start = 0
        for entry in self.shards:
            if entry.source_start_index != expected_start:
                raise ValueError("Shard source ranges are not contiguous.")
            expected_start = entry.source_end_index
        if expected_start != self.record_count:
            raise ValueError("Catalog source ranges do not cover the complete record count.")
        return self


def validate_embedding_catalog_files(catalog: EmbeddingRunCatalog) -> None:
    """Validate catalog entries against files on disk."""
    total_records = 0
    chunk_ids: set[str] = set()
    for entry in catalog.shards:
        root = Path(entry.shard_root)
        header_path = Path(entry.header_path)
        records_path = Path(entry.records_path)
        manifest_path = Path(entry.manifest_path)

        if root != header_path.parent:
            raise ValueError("Header is outside its declared shard root.")

        header = read_json_model(header_path, EmbeddingShardHeader)
        manifest = read_json_model(manifest_path, ArtifactSidecarManifest)

        if header.shard_id != entry.shard_id:
            raise ValueError("Catalog shard ID does not match header.")
        if header.embedding_config_hash != catalog.embedding_config_hash:
            raise ValueError("Shard configuration hash does not match the catalog.")
        if sha256_file(header_path) != entry.header_sha256:
            raise ValueError("Catalog header checksum mismatch.")
        if sha256_file(records_path) != entry.records_sha256:
            raise ValueError("Catalog records checksum mismatch.")
        if sha256_file(manifest_path) != entry.manifest_sha256:
            raise ValueError("Catalog manifest checksum mismatch.")
        if manifest.artifact_sha256 != entry.records_sha256:
            raise ValueError("Shard sidecar checksum does not match the catalog.")

        records = tuple(iter_embedding_records(records_path))
        if len(records) != entry.record_count:
            raise ValueError("Catalog record count does not match the shard.")

        for record in records:
            if record.chunk_id in chunk_ids:
                raise ValueError("Catalog contains duplicate chunk IDs.")
            if record.embedding_config_hash != catalog.embedding_config_hash:
                raise ValueError("Embedding record configuration hash does not match the catalog.")
            chunk_ids.add(record.chunk_id)
        total_records += len(records)

    if total_records != catalog.record_count:
        raise ValueError("Validated shard records do not match the catalog total.")
