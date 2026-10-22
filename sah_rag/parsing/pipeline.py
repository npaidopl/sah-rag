"""Streaming canonical-document parsing pipeline."""

from __future__ import annotations

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
from sah_rag.artifacts.models import ParsedDocument
from sah_rag.canonical.loaders import (
    iter_canonical_documents,
)
from sah_rag.canonical.paths import (
    resolve_dataset_paths,
)
from sah_rag.parsing.deterministic import (
    DeterministicStructuralParser,
)


@dataclass(frozen=True, slots=True)
class ParsingRunSummary:
    """Summary of one streaming parsing run."""

    dataset_id: str
    document_count: int
    parent_count: int
    empty_document_count: int
    output_path: Path
    manifest_path: Path | None = None


def parse_canonical_dataset(
    *,
    dataset_id: str,
    parser: DeterministicStructuralParser,
    output_path: Path,
    limit: int | None = None,
    overwrite: bool = False,
    manifest_path: Path | None = None,
    execution_profile: str | None = None,
    experiment_id: str | None = None,
    manifest_metadata: dict[str, Any] | None = None,
) -> ParsingRunSummary:
    """Parse canonical documents and stream artifacts to JSONL."""

    if limit is not None and limit < 1:
        raise ValueError(
            "Parsing limit must be at least 1."
        )

    if (
        manifest_path is not None
        and manifest_path.resolve() == output_path.resolve()
    ):
        raise ValueError(
            "Manifest path must differ from the artifact path."
        )

    document_count = 0
    parent_count = 0
    empty_document_count = 0

    with AtomicJsonlWriter(
        output_path,
        overwrite=overwrite,
    ) as writer:
        for document in iter_canonical_documents(
            dataset_id
        ):
            if (
                limit is not None
                and document_count >= limit
            ):
                break

            parsed_document = parser.parse(document)

            _validate_parsed_source_mapping(
                parsed_document=parsed_document,
                canonical_text=document.text_normalized,
            )

            writer.write(parsed_document)

            document_count += 1
            parent_count += len(
                parsed_document.parents
            )

            if not parsed_document.parents:
                empty_document_count += 1

    if manifest_path is not None:
        try:
            _write_parsing_manifest(
                dataset_id=dataset_id,
                parser=parser,
                output_path=output_path,
                manifest_path=manifest_path,
                document_count=document_count,
                parent_count=parent_count,
                empty_document_count=(
                    empty_document_count
                ),
                limit=limit,
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

    return ParsingRunSummary(
        dataset_id=dataset_id,
        document_count=document_count,
        parent_count=parent_count,
        empty_document_count=empty_document_count,
        output_path=output_path,
        manifest_path=manifest_path,
    )


def _write_parsing_manifest(
    *,
    dataset_id: str,
    parser: DeterministicStructuralParser,
    output_path: Path,
    manifest_path: Path,
    document_count: int,
    parent_count: int,
    empty_document_count: int,
    limit: int | None,
    overwrite: bool,
    execution_profile: str | None,
    experiment_id: str | None,
    manifest_metadata: dict[str, Any] | None,
) -> ArtifactSidecarManifest:
    """Create and publish the parsed-artifact sidecar."""

    dataset_paths = resolve_dataset_paths(dataset_id)

    canonical_input = build_input_reference(
        artifact_type="canonical_documents",
        path=dataset_paths.documents,
    )

    component = ArtifactComponentReference(
        component_type="parser",
        component_id=parser.config.parser_id,
        component_version=parser.config.parser_version,
        configuration_hash=parser.config_hash,
        configuration=parser.config.model_dump(
            mode="json"
        ),
    )

    metadata: dict[str, Any] = {
        "document_count": document_count,
        "parent_count": parent_count,
        "empty_document_count": (
            empty_document_count
        ),
        "document_limit": limit,
    }

    if manifest_metadata is not None:
        metadata.update(manifest_metadata)

    manifest = build_jsonl_sidecar_manifest(
        artifact_type="parsed_documents",
        artifact_path=output_path,
        component=component,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        inputs=(canonical_input,),
        code_revision=get_git_revision(),
        execution_profile=execution_profile,
        metadata=metadata,
    )

    if manifest.record_count != document_count:
        raise RuntimeError(
            "Parsed manifest record count does not match "
            "the parsing-run summary."
        )

    write_json_model_atomic(
        manifest_path,
        manifest,
        overwrite=overwrite,
    )

    return manifest


def _validate_parsed_source_mapping(
    *,
    parsed_document: ParsedDocument,
    canonical_text: str,
) -> None:
    """Verify every parent maps exactly to its canonical slice."""

    for parent in parsed_document.parents:
        span = parent.normalized_char_span

        source_slice = canonical_text[
            span.start:span.end
        ]

        if source_slice != parent.text_normalized:
            raise ValueError(
                "Semantic-parent text does not match "
                "its canonical normalized-text slice: "
                f"{parent.parent_id}"
            )
