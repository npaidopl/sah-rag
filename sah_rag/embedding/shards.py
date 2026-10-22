"""Writing and validation of embedding shard artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sah_rag.artifacts import (
    ArtifactComponentReference,
    ArtifactSidecarManifest,
    build_input_reference,
    build_jsonl_sidecar_manifest,
    get_git_revision,
    read_json_model,
    write_json_model_atomic,
)
from sah_rag.artifacts.fingerprints import count_nonempty_lines
from sah_rag.artifacts.jsonl import AtomicJsonlWriter
from sah_rag.canonical.jsonl import iter_jsonl
from sah_rag.embedding.models import (
    ChunkEmbeddingRecord,
    EmbeddingKind,
    EmbeddingShardHeader,
)


@dataclass(frozen=True, slots=True)
class EmbeddingShardPaths:
    """Resolved files belonging to one embedding shard."""

    root: Path
    header: Path
    records: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class EmbeddingShardWriteSummary:
    """Summary of one completed shard write."""

    shard_id: str
    record_count: int
    paths: EmbeddingShardPaths


def resolve_embedding_shard_paths(
    root: Path,
) -> EmbeddingShardPaths:
    """Resolve the conventional files inside a shard directory."""

    return EmbeddingShardPaths(
        root=root,
        header=root / "header.json",
        records=root / "records.jsonl",
        manifest=root / "shard.manifest.json",
    )


def iter_embedding_records(
    path: Path,
) -> Iterator[ChunkEmbeddingRecord]:
    """Stream validated embedding records from JSONL."""

    for raw_record in iter_jsonl(path):
        yield ChunkEmbeddingRecord.model_validate(raw_record)






def _load_source_chunk_ids(
    path: Path,
) -> set[str]:
    """Load source chunk IDs for shard membership validation."""

    chunk_ids: set[str] = set()

    for raw_record in iter_jsonl(path):
        chunk_id = raw_record.get("chunk_id")

        if not isinstance(chunk_id, str):
            raise ValueError(
                "Source chunk record has no valid chunk_id."
            )

        if not chunk_id.strip():
            raise ValueError(
                "Source chunk IDs cannot be empty."
            )

        if chunk_id in chunk_ids:
            raise ValueError(
                "Source chunk artifact contains duplicate "
                f"chunk_id: {chunk_id}"
            )

        chunk_ids.add(chunk_id)

    return chunk_ids




def write_embedding_shard(
    *,
    root: Path,
    header: EmbeddingShardHeader,
    records: Iterable[ChunkEmbeddingRecord],
    source_chunks_path: Path,
    source_chunks_manifest_path: Path | None = None,
    overwrite: bool = False,
    execution_profile: str | None = None,
    experiment_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EmbeddingShardWriteSummary:
    """Atomically publish one validated embedding shard."""

    paths = resolve_embedding_shard_paths(root)

    if paths.root.exists() and not paths.root.is_dir():
        raise NotADirectoryError(f"Shard root is not a directory: {paths.root}")

    paths.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not overwrite:
        existing_paths = [
            path
            for path in (
                paths.header,
                paths.records,
                paths.manifest,
            )
            if path.exists()
        ]

        if existing_paths:
            raise FileExistsError(
                "Embedding shard outputs already exist: "
                + ", ".join(str(path) for path in existing_paths)
            )

    source_chunk_ids = _load_source_chunk_ids(
        source_chunks_path
    )

    try:
        record_count = _write_embedding_records(
            path=paths.records,
            header=header,
            records=records,
            overwrite=overwrite,
            source_chunk_ids=source_chunk_ids,
        )

        write_json_model_atomic(
            paths.header,
            header,
            overwrite=overwrite,
        )

        _write_shard_manifest(
            paths=paths,
            header=header,
            record_count=record_count,
            source_chunks_path=source_chunks_path,
            source_chunks_manifest_path=(source_chunks_manifest_path),
            overwrite=overwrite,
            execution_profile=execution_profile,
            experiment_id=experiment_id,
            metadata=metadata,
            source_chunk_count=len(
                source_chunk_ids
            ),
        )
    except BaseException:
        paths.header.unlink(missing_ok=True)
        paths.records.unlink(missing_ok=True)
        paths.manifest.unlink(missing_ok=True)
        raise

    return EmbeddingShardWriteSummary(
        shard_id=header.shard_id,
        record_count=record_count,
        paths=paths,
    )


def _write_embedding_records(
    *,
    path: Path,
    header: EmbeddingShardHeader,
    records: Iterable[ChunkEmbeddingRecord],
    overwrite: bool,
    source_chunk_ids: set[str],
) -> int:
    """Validate and write all records in one shard."""

    record_count = 0
    chunk_ids: set[str] = set()

    with AtomicJsonlWriter(
        path,
        overwrite=overwrite,
    ) as writer:
        for record in records:
            _validate_record_against_header(
                record=record,
                header=header,
                source_chunk_ids=source_chunk_ids,
            )

            if record.chunk_id in chunk_ids:
                raise ValueError(f"Embedding shard contains duplicate chunk_id: {record.chunk_id}")

            if record_count >= header.maximum_records:
                raise ValueError("Embedding shard exceeds maximum_records.")

            chunk_ids.add(record.chunk_id)
            writer.write(record)
            record_count += 1

    return record_count


def _validate_record_against_header(
    *,
    record: ChunkEmbeddingRecord,
    header: EmbeddingShardHeader,
    source_chunk_ids: set[str],
) -> None:
    """Validate one embedding record against shard metadata."""

    expected_values = {
        "dataset": (
            record.dataset,
            header.dataset,
        ),
        "model_id": (
            record.model_id,
            header.model_id,
        ),
        "model_revision": (
            record.model_revision,
            header.model_revision,
        ),
        "tokenizer_id": (
            record.tokenizer_id,
            header.tokenizer_id,
        ),
        "embedding_config_hash": (
            record.embedding_config_hash,
            header.embedding_config_hash,
        ),
    }

    for field_name, (
        record_value,
        header_value,
    ) in expected_values.items():
        if record_value != header_value:
            raise ValueError(f"Embedding record {field_name} does not match the shard header.")

    dense_expected = header.embedding_kind in {
        EmbeddingKind.DENSE,
        EmbeddingKind.HYBRID,
    }

    sparse_expected = header.embedding_kind in {
        EmbeddingKind.LEARNED_SPARSE,
        EmbeddingKind.HYBRID,
    }

    if dense_expected != (record.dense_vector is not None):
        raise ValueError("Embedding record dense representation does not match the shard kind.")

    if sparse_expected != (record.sparse_vector is not None):
        raise ValueError("Embedding record sparse representation does not match the shard kind.")

    if dense_expected:
        if record.dense_dimension != header.dense_dimension:
            raise ValueError("Embedding record dense dimension does not match the shard header.")

        if record.dense_precision != header.dense_precision:
            raise ValueError("Embedding record dense precision does not match the shard header.")

        if record.dense_normalized != header.dense_normalized:
            raise ValueError(
                "Embedding record normalization metadata does not match the shard header."
            )

    if record.chunk_id not in source_chunk_ids:
        raise ValueError(
            "Embedding record references an unknown "
            f"source chunk_id: {record.chunk_id}"
        )



def _write_shard_manifest(
    *,
    source_chunk_count: int,
    paths: EmbeddingShardPaths,
    header: EmbeddingShardHeader,
    record_count: int,
    source_chunks_path: Path,
    source_chunks_manifest_path: Path | None,
    overwrite: bool,
    execution_profile: str | None,
    experiment_id: str | None,
    metadata: dict[str, Any] | None,
) -> ArtifactSidecarManifest:
    """Build and publish an embedding-shard sidecar."""


    source_manifest = None
    source_record_count = None
    if source_chunks_manifest_path is not None:
        source_manifest = read_json_model(
            source_chunks_manifest_path,
            ArtifactSidecarManifest,
        )
        source_record_count = source_manifest.record_count
    else:
        source_record_count = count_nonempty_lines(source_chunks_path)

    source_chunks = build_input_reference(
        artifact_type="child_chunks",
        path=source_chunks_path,
        record_count=source_record_count,
    )

    if source_manifest is not None:
        if source_manifest.artifact_sha256 != source_chunks.sha256:
            raise ValueError("Source chunk checksum does not match its supplied manifest.")

        if source_manifest.artifact_size_bytes != source_chunks.size_bytes:
            raise ValueError("Source chunk size does not match its supplied manifest.")

        if source_manifest.record_count != source_chunks.record_count:
            raise ValueError("Source chunk record count does not match its supplied manifest.")

    header_reference = build_input_reference(
        artifact_type="embedding_shard_header",
        path=paths.header,
        record_count=1,
    )

    component = ArtifactComponentReference(
        component_type="embedder",
        component_id=header.model_id,
        component_version=header.model_revision,
        configuration_hash=(header.embedding_config_hash),
        configuration={
            "embedding_kind": (header.embedding_kind.value),
            "dense_dimension": (header.dense_dimension),
            "dense_precision": (
                header.dense_precision.value if header.dense_precision is not None else None
            ),
            "dense_normalized": (header.dense_normalized),
            "tokenizer_id": header.tokenizer_id,
            "maximum_records": (header.maximum_records),
        },
    )

    manifest_metadata: dict[str, Any] = {
        "shard_id": header.shard_id,
        "shard_index": header.shard_index,
        "header_path": str(paths.header.resolve()),
        "header_sha256": (header_reference.sha256),
        "header_size_bytes": (header_reference.size_bytes),
        "source_chunk_count": source_chunk_count,
        "embedded_chunk_count": record_count,
        "complete_source_coverage": (
            record_count == source_chunk_count
        ),


    }

    if source_chunks_manifest_path is not None:
        manifest_metadata["source_chunks_manifest_path"] = str(
            source_chunks_manifest_path.resolve()
        )

    if metadata is not None:
        manifest_metadata.update(metadata)

    manifest = build_jsonl_sidecar_manifest(
        artifact_type="embedding_shard",
        artifact_path=paths.records,
        component=component,
        dataset_id=header.dataset,
        experiment_id=experiment_id,
        inputs=(
            source_chunks,
            header_reference,
        ),
        code_revision=get_git_revision(),
        execution_profile=execution_profile,
        metadata=manifest_metadata,
    )

    if manifest.record_count != record_count:
        raise RuntimeError(
            "Embedding shard manifest record count does not match the writer summary."
        )

    write_json_model_atomic(
        paths.manifest,
        manifest,
        overwrite=overwrite,
    )

    return manifest
