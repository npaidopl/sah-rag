"""Tests for artifact sidecar manifests."""

from __future__ import annotations

import json
from pathlib import Path

from rag_system.artifacts import (
    ArtifactComponentReference,
    ArtifactSidecarManifest,
    build_input_reference,
    build_jsonl_sidecar_manifest,
    read_json_model,
    write_json_model_atomic,
)


def make_component() -> ArtifactComponentReference:
    """Create a sample producer reference."""

    return ArtifactComponentReference(
        component_type="chunker",
        component_id="hierarchical_token_chunker",
        component_version="0.3.0",
        configuration_hash="chunker-hash",
        configuration={
            "chunk_size": 256,
            "chunk_overlap": 32,
        },
    )


def test_build_jsonl_sidecar_manifest(
    tmp_path: Path,
) -> None:
    """A JSONL artifact should receive exact metadata."""

    artifact_path = tmp_path / "chunks.jsonl"

    artifact_path.write_text(
        '{"chunk_id":"a"}\n'
        '{"chunk_id":"b"}\n',
        encoding="utf-8",
    )

    manifest = build_jsonl_sidecar_manifest(
        artifact_type="child_chunks",
        artifact_path=artifact_path,
        component=make_component(),
        dataset_id="synthetic",
        code_revision="abc123",
        execution_profile="local_cpu",
    )

    assert manifest.record_count == 2
    assert manifest.artifact_size_bytes > 0
    assert len(manifest.artifact_sha256) == 64
    assert manifest.dataset_id == "synthetic"


def test_input_reference_is_fingerprinted(
    tmp_path: Path,
) -> None:
    """Consumed inputs should preserve checksums."""

    input_path = tmp_path / "parsed.jsonl"

    input_path.write_text(
        '{"doc_id":"doc-1"}\n',
        encoding="utf-8",
    )

    reference = build_input_reference(
        artifact_type="parsed_documents",
        path=input_path,
        record_count=1,
    )

    assert len(reference.sha256) == 64
    assert reference.record_count == 1
    assert reference.size_bytes > 0


def test_manifest_round_trip(
    tmp_path: Path,
) -> None:
    """Written sidecar manifests should validate on read."""

    artifact_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "chunks.manifest.json"

    artifact_path.write_text(
        '{"chunk_id":"a"}\n',
        encoding="utf-8",
    )

    manifest = build_jsonl_sidecar_manifest(
        artifact_type="child_chunks",
        artifact_path=artifact_path,
        component=make_component(),
    )

    write_json_model_atomic(
        manifest_path,
        manifest,
    )

    loaded = read_json_model(
        manifest_path,
        ArtifactSidecarManifest,
    )

    assert loaded == manifest

    raw_manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert raw_manifest["record_count"] == 1


def test_manifest_writer_rejects_existing_file(
    tmp_path: Path,
) -> None:
    """Existing manifests require explicit overwrite."""

    artifact_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "chunks.manifest.json"

    artifact_path.write_text(
        '{"chunk_id":"a"}\n',
        encoding="utf-8",
    )

    manifest = build_jsonl_sidecar_manifest(
        artifact_type="child_chunks",
        artifact_path=artifact_path,
        component=make_component(),
    )

    write_json_model_atomic(
        manifest_path,
        manifest,
    )

    try:
        write_json_model_atomic(
            manifest_path,
            manifest,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError(
            "Expected FileExistsError."
        )


def test_general_file_sidecar_uses_explicit_count(tmp_path: Path) -> None:
    """General JSON sidecars should use semantic counts."""
    from rag_system.artifacts import build_file_sidecar_manifest

    artifact_path = tmp_path / "catalog.json"
    artifact_path.write_text('{\n  "run_id": "run-001",\n  "shards": []\n}\n', encoding="utf-8")

    manifest = build_file_sidecar_manifest(
        artifact_type="embedding_run_catalog",
        artifact_path=artifact_path,
        record_count=1,
        component=make_component(),
    )

    assert manifest.record_count == 1
    assert manifest.artifact_size_bytes == artifact_path.stat().st_size
    assert len(manifest.artifact_sha256) == 64
