"""Streaming pipeline from parsed documents to child chunks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sah_rag.artifacts import (
    ArtifactComponentReference,
    ArtifactSidecarManifest,
    build_input_reference,
    build_jsonl_sidecar_manifest,
    get_git_revision,
    write_json_model_atomic,
)
from sah_rag.artifacts.jsonl import AtomicJsonlWriter
from sah_rag.artifacts.models import (
    ChildChunk,
    ParsedDocument,
)
from sah_rag.canonical.jsonl import iter_jsonl
from sah_rag.chunking.hierarchical import (
    HierarchicalChildChunker,
)


@dataclass(frozen=True, slots=True)
class ChunkingRunSummary:
    """Summary of one parsed-document chunking run."""

    parsed_document_count: int
    parent_count: int
    chunk_count: int
    parents_without_chunks: int
    output_path: Path
    manifest_path: Path | None = None


def iter_parsed_documents(
    path: Path,
) -> Iterator[ParsedDocument]:
    """Stream validated parsed documents from JSONL."""

    for record in iter_jsonl(path):
        yield ParsedDocument.model_validate(record)


def chunk_parsed_artifacts(
    *,
    input_path: Path,
    output_path: Path,
    chunker: HierarchicalChildChunker,
    overwrite: bool = False,
    manifest_path: Path | None = None,
    input_manifest_path: Path | None = None,
    dataset_id: str | None = None,
    execution_profile: str | None = None,
    experiment_id: str | None = None,
    manifest_metadata: dict[str, Any] | None = None,
) -> ChunkingRunSummary:
    """Stream parsed documents into child-chunk JSONL."""

    if (
        manifest_path is not None
        and manifest_path.resolve() == output_path.resolve()
    ):
        raise ValueError(
            "Manifest path must differ from the artifact path."
        )

    parsed_document_count = 0
    parent_count = 0
    chunk_count = 0
    parents_without_chunks = 0
    detected_dataset_id: str | None = None

    with AtomicJsonlWriter(
        output_path,
        overwrite=overwrite,
    ) as writer:
        for parsed_document in iter_parsed_documents(
            input_path
        ):
            if detected_dataset_id is None:
                detected_dataset_id = (
                    parsed_document.dataset
                )
            elif (
                parsed_document.dataset
                != detected_dataset_id
            ):
                raise ValueError(
                    "Chunking input contains multiple datasets."
                )

            chunks = chunker.chunk_document(
                parsed_document
            )

            parsed_document_count += 1
            parent_count += len(
                parsed_document.parents
            )
            chunk_count += len(chunks)

            parents_with_chunks = {
                chunk.parent_id
                for chunk in chunks
            }

            parents_without_chunks += sum(
                parent.parent_id
                not in parents_with_chunks
                for parent in parsed_document.parents
            )

            for chunk in chunks:
                _validate_chunk_source_mapping(
                    parsed_document=parsed_document,
                    chunk=chunk,
                )

                writer.write(chunk)

    effective_dataset_id = (
        dataset_id
        if dataset_id is not None
        else detected_dataset_id
    )

    if (
        dataset_id is not None
        and detected_dataset_id is not None
        and dataset_id != detected_dataset_id
    ):
        output_path.unlink(
            missing_ok=True
        )
        raise ValueError(
            "Configured dataset_id does not match "
            "the parsed artifact."
        )

    if manifest_path is not None:
        try:
            _write_chunking_manifest(
                input_path=input_path,
                input_manifest_path=input_manifest_path,
                output_path=output_path,
                manifest_path=manifest_path,
                chunker=chunker,
                dataset_id=effective_dataset_id,
                parsed_document_count=(
                    parsed_document_count
                ),
                parent_count=parent_count,
                chunk_count=chunk_count,
                parents_without_chunks=(
                    parents_without_chunks
                ),
                overwrite=overwrite,
                execution_profile=execution_profile,
                experiment_id=experiment_id,
                manifest_metadata=manifest_metadata,
            )
        except BaseException:
            output_path.unlink(
                missing_ok=True
            )
            manifest_path.unlink(
                missing_ok=True
            )
            raise

    return ChunkingRunSummary(
        parsed_document_count=parsed_document_count,
        parent_count=parent_count,
        chunk_count=chunk_count,
        parents_without_chunks=parents_without_chunks,
        output_path=output_path,
        manifest_path=manifest_path,
    )


def _write_chunking_manifest(
    *,
    input_path: Path,
    input_manifest_path: Path | None,
    output_path: Path,
    manifest_path: Path,
    chunker: HierarchicalChildChunker,
    dataset_id: str | None,
    parsed_document_count: int,
    parent_count: int,
    chunk_count: int,
    parents_without_chunks: int,
    overwrite: bool,
    execution_profile: str | None,
    experiment_id: str | None,
    manifest_metadata: dict[str, Any] | None,
) -> ArtifactSidecarManifest:
    """Create and publish the child-artifact sidecar."""

    parsed_input = build_input_reference(
        artifact_type="parsed_documents",
        path=input_path,
        record_count=parsed_document_count,
    )

    if input_manifest_path is not None:
        from sah_rag.artifacts import (
            ArtifactSidecarManifest as SidecarModel,
        )
        from sah_rag.artifacts import (
            read_json_model,
        )

        input_manifest = read_json_model(
            input_manifest_path,
            SidecarModel,
        )

        if (
            input_manifest.artifact_sha256
            != parsed_input.sha256
        ):
            raise ValueError(
                "Parsed input checksum does not match "
                "its supplied sidecar manifest."
            )

        if (
            input_manifest.artifact_size_bytes
            != parsed_input.size_bytes
        ):
            raise ValueError(
                "Parsed input size does not match "
                "its supplied sidecar manifest."
            )

        if (
            input_manifest.record_count
            != parsed_input.record_count
        ):
            raise ValueError(
                "Parsed input record count does not match "
                "its supplied sidecar manifest."
            )

    component = ArtifactComponentReference(
        component_type="chunker",
        component_id=chunker.config.chunker_id,
        component_version=(
            chunker.config.chunker_version
        ),
        configuration_hash=chunker.config_hash,
        configuration={
            **chunker.config.model_dump(
                mode="json"
            ),
            "tokenizer_id": (
                chunker.tokenizer.tokenizer_id
            ),
            "special_tokens_count": (
                chunker.tokenizer.special_tokens_count
            ),
            "content_chunk_size": (
                chunker.content_chunk_size
            ),
        },
    )

    metadata: dict[str, Any] = {
        "parsed_document_count": (
            parsed_document_count
        ),
        "parent_count": parent_count,
        "chunk_count": chunk_count,
        "parents_without_chunks": (
            parents_without_chunks
        ),
    }

    if input_manifest_path is not None:
        metadata["input_manifest_path"] = str(
            input_manifest_path.resolve()
        )

    if manifest_metadata is not None:
        metadata.update(manifest_metadata)

    manifest = build_jsonl_sidecar_manifest(
        artifact_type="child_chunks",
        artifact_path=output_path,
        component=component,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        inputs=(parsed_input,),
        code_revision=get_git_revision(),
        execution_profile=execution_profile,
        metadata=metadata,
    )

    if manifest.record_count != chunk_count:
        raise RuntimeError(
            "Chunk manifest record count does not match "
            "the chunking-run summary."
        )

    write_json_model_atomic(
        manifest_path,
        manifest,
        overwrite=overwrite,
    )

    return manifest


def _validate_chunk_source_mapping(
    *,
    parsed_document: ParsedDocument,
    chunk: ChildChunk,
) -> None:
    """Verify a chunk against its semantic-parent source."""

    parents = {
        parent.parent_id: parent
        for parent in parsed_document.parents
    }

    parent = parents.get(chunk.parent_id)

    if parent is None:
        raise ValueError(
            f"Unknown parent for chunk {chunk.chunk_id}."
        )

    relative_start = (
        chunk.normalized_char_span.start
        - parent.normalized_char_span.start
    )
    relative_end = (
        chunk.normalized_char_span.end
        - parent.normalized_char_span.start
    )

    source_slice = parent.text_normalized[
        relative_start:relative_end
    ]

    if source_slice != chunk.text_normalized:
        raise ValueError(
            "Chunk text does not match its parent source slice: "
            f"{chunk.chunk_id}"
        )
