"""Integration test for generated artifact sidecars."""

from __future__ import annotations

from pathlib import Path

from rag_system.artifacts import (
    ArtifactComponentReference,
    ArtifactSidecarManifest,
    build_input_reference,
    build_jsonl_sidecar_manifest,
    read_json_model,
    write_json_model_atomic,
)
from rag_system.artifacts.jsonl import (
    write_jsonl_atomic,
)


def test_jsonl_artifact_and_sidecar(
    tmp_path: Path,
) -> None:
    """A generated JSONL artifact should be fully traceable."""

    artifact_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "chunks.manifest.json"

    record_count = write_jsonl_atomic(
        artifact_path,
        [
            {"chunk_id": "chunk-a"},
            {"chunk_id": "chunk-b"},
        ],
    )

    component = ArtifactComponentReference(
        component_type="chunker",
        component_id="test_chunker",
        component_version="0.1.0",
        configuration_hash="test-hash",
        configuration={},
    )

    manifest = build_jsonl_sidecar_manifest(
        artifact_type="child_chunks",
        artifact_path=artifact_path,
        component=component,
        dataset_id="synthetic",
        metadata={
            "reported_record_count": record_count,
        },
    )

    write_json_model_atomic(
        manifest_path,
        manifest,
    )

    loaded = read_json_model(
        manifest_path,
        ArtifactSidecarManifest,
    )

    assert loaded.record_count == record_count
    assert loaded.artifact_path == str(
        artifact_path.resolve()
    )
    assert loaded.artifact_size_bytes == (
        artifact_path.stat().st_size
    )


def test_downstream_input_matches_upstream_manifest(
    tmp_path: Path,
) -> None:
    """A downstream sidecar must reference the exact upstream artifact."""

    parsed_path = tmp_path / "parsed.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"

    write_jsonl_atomic(
        parsed_path,
        [{"doc_id": "doc-001"}],
    )

    write_jsonl_atomic(
        chunks_path,
        [{"chunk_id": "chunk-001"}],
    )

    parser_component = ArtifactComponentReference(
        component_type="parser",
        component_id="test-parser",
        component_version="0.1.0",
        configuration_hash="parser-hash",
        configuration={},
    )

    parsed_manifest = build_jsonl_sidecar_manifest(
        artifact_type="parsed_documents",
        artifact_path=parsed_path,
        component=parser_component,
    )

    parsed_reference = build_input_reference(
        artifact_type="parsed_documents",
        path=parsed_path,
        record_count=parsed_manifest.record_count,
    )

    chunker_component = ArtifactComponentReference(
        component_type="chunker",
        component_id="test-chunker",
        component_version="0.1.0",
        configuration_hash="chunker-hash",
        configuration={},
    )

    chunks_manifest = build_jsonl_sidecar_manifest(
        artifact_type="child_chunks",
        artifact_path=chunks_path,
        component=chunker_component,
        inputs=(parsed_reference,),
    )

    assert (
        chunks_manifest.inputs[0].sha256
        == parsed_manifest.artifact_sha256
    )
    assert (
        chunks_manifest.inputs[0].size_bytes
        == parsed_manifest.artifact_size_bytes
    )
    assert (
        chunks_manifest.inputs[0].record_count
        == parsed_manifest.record_count
    )
