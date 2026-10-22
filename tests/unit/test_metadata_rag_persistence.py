"""Tests for baselines.metadata_rag_persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from baselines.metadata_rag_adapter import (
    build_indexed_documents,
)
from baselines.metadata_rag_persistence import (
    ARTIFACT_TYPE,
    ARTIFACT_VERSION,
    CHECKSUMS_FILE,
    DOCUMENTS_FILE,
    MANIFEST_FILE,
    MetadataRAGPersistenceError,
    read_state,
    verify_checksums,
    write_state,
)
from baselines.metadata_rag_state import (
    MetadataRAGState,
    build_state,
)
from sah_rag.canonical.loaders import iter_validated_models
from sah_rag.canonical.models import CanonicalDocument

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    REPO_ROOT
    / "processed"
    / "canonical"
    / "infrastructure-pilot-synthetic-v1"
)
FIXTURE_DATASET = "infrastructure-pilot-synthetic-v1"


def _fixture_state() -> MetadataRAGState:
    docs = tuple(
        iter_validated_models(
            FIXTURE_ROOT / "documents.jsonl", CanonicalDocument
        )
    )
    indexed = build_indexed_documents(docs, dataset_id=FIXTURE_DATASET)
    return build_state(indexed, dataset_id=FIXTURE_DATASET)


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)

    assert (tmp_path / DOCUMENTS_FILE).is_file()
    assert (tmp_path / MANIFEST_FILE).is_file()
    assert (tmp_path / CHECKSUMS_FILE).is_file()

    reopened = read_state(tmp_path)
    assert reopened.dataset_id == state.dataset_id
    assert reopened.state_version == state.state_version
    assert len(reopened.documents) == len(state.documents)
    for original, observed in zip(
        state.documents, reopened.documents, strict=True
    ):
        assert original == observed


def test_metadata_record_ids_survive_round_trip(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)
    reopened = read_state(tmp_path)
    original_ids = [d.metadata_record_id for d in state.documents]
    reopened_ids = [d.metadata_record_id for d in reopened.documents]
    assert original_ids == reopened_ids


def test_writes_are_byte_deterministic(tmp_path: Path) -> None:
    state = _fixture_state()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    hashes_a = write_state(state, out_a)
    hashes_b = write_state(state, out_b)
    assert hashes_a == hashes_b
    assert (out_a / DOCUMENTS_FILE).read_bytes() == (
        out_b / DOCUMENTS_FILE
    ).read_bytes()
    assert (out_a / MANIFEST_FILE).read_bytes() == (
        out_b / MANIFEST_FILE
    ).read_bytes()


def test_documents_jsonl_is_lf_terminated_compact(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)
    payload = (tmp_path / DOCUMENTS_FILE).read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r" not in payload
    assert not payload.startswith(b"\xef\xbb\xbf")
    text = payload.decode("utf-8")
    for line in text.splitlines():
        obj = json.loads(line)
        assert isinstance(obj, dict)
        assert obj["record_type"] == "metadata_rag_indexed_document"


def test_manifest_envelope_fields(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text("utf-8"))
    assert manifest["artifact_type"] == ARTIFACT_TYPE
    assert manifest["artifact_version"] == ARTIFACT_VERSION
    assert manifest["dataset_id"] == FIXTURE_DATASET
    assert manifest["document_count"] == 12
    assert manifest["generation_enabled"] is False
    assert manifest["retrieval_execution_enabled"] is False
    assert manifest["evaluation_execution_enabled"] is False
    assert "documents_sha256" in manifest


def test_verify_checksums_passes_on_clean_directory(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)
    verified = verify_checksums(tmp_path)
    assert set(verified.keys()) == {DOCUMENTS_FILE, MANIFEST_FILE}


def test_read_state_detects_tampered_documents(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)
    target = tmp_path / DOCUMENTS_FILE
    payload = target.read_bytes()
    tampered = payload.replace(b"pilot-doc-01", b"pilot-doc-99", 1)
    assert tampered != payload
    target.write_bytes(tampered)
    with pytest.raises(
        MetadataRAGPersistenceError, match="checksum mismatch"
    ):
        read_state(tmp_path)


def test_read_state_detects_tampered_manifest(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)
    target = tmp_path / MANIFEST_FILE
    manifest = json.loads(target.read_text("utf-8"))
    manifest["document_count"] = 99
    target.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        MetadataRAGPersistenceError, match="checksum mismatch"
    ):
        read_state(tmp_path)


def test_read_state_rejects_wrong_artifact_type(tmp_path: Path) -> None:
    state = _fixture_state()
    write_state(state, tmp_path)
    target = tmp_path / MANIFEST_FILE
    manifest = json.loads(target.read_text("utf-8"))
    manifest["artifact_type"] = "something_else"
    target.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    checks = json.loads((tmp_path / CHECKSUMS_FILE).read_text("utf-8"))
    checks[MANIFEST_FILE] = hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    (tmp_path / CHECKSUMS_FILE).write_text(
        json.dumps(checks, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(
        MetadataRAGPersistenceError, match="artifact_type mismatch"
    ):
        read_state(tmp_path)


def test_read_state_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(MetadataRAGPersistenceError):
        read_state(tmp_path / "nope")


def test_write_state_rejects_non_state(tmp_path: Path) -> None:
    with pytest.raises(MetadataRAGPersistenceError):
        write_state("not-a-state", tmp_path)  # type: ignore[arg-type]
