"""Deterministic disk persistence for the Metadata-RAG comparator.

Writes the Metadata-RAG state to a flat, checksum-bound, JSONL-based
artifact. The artifact is:

    documents.jsonl   one MetadataRAGIndexedDocument per line, sorted
                      by document_order, compact JSON with sorted keys
    manifest.json     dataset and version envelope
    checksums.json    SHA-256 of every sibling artifact

Every file is UTF-8 without BOM and uses LF line endings. Writes are
atomic: content is written to a temporary sibling and then renamed
over the target. Reopening a persisted state is validated against
the on-disk checksums and against the state-model invariants.

Frozen specification:
    baseline_audit/metadata_rag_clean_room_spec.yaml
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, cast

from baselines.metadata_rag_adapter import (
    ADAPTER_VERSION,
    MetadataRAGIndexedDocument,
)
from baselines.metadata_rag_metadata import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
)
from baselines.metadata_rag_state import (
    STATE_VERSION,
    MetadataRAGState,
)

ARTIFACT_TYPE: Final[str] = "metadata_rag_state"
ARTIFACT_VERSION: Final[str] = "1.0"

DOCUMENTS_FILE: Final[str] = "documents.jsonl"
MANIFEST_FILE: Final[str] = "manifest.json"
CHECKSUMS_FILE: Final[str] = "checksums.json"

_CHUNK_SIZE: Final[int] = 1 << 20


class MetadataRAGPersistenceError(ValueError):
    """Raised when a persisted state is missing or inconsistent."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, target)


def _encode_json(obj: Any) -> bytes:
    """Deterministic compact JSON with LF and no BOM."""
    text = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return (text + "\n").encode("utf-8")


def _encode_jsonl(lines: list[Any]) -> bytes:
    """Deterministic compact JSONL with LF, no BOM, trailing newline."""
    if not lines:
        raise MetadataRAGPersistenceError("jsonl cannot be empty")
    parts: list[str] = []
    for obj in lines:
        parts.append(
            json.dumps(
                obj,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
    return ("\n".join(parts) + "\n").encode("utf-8")


def _serialize_document(
    doc: MetadataRAGIndexedDocument,
) -> dict[str, Any]:
    payload = doc.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise MetadataRAGPersistenceError(
            "document.model_dump did not return a mapping"
        )
    return payload


def build_manifest(
    state: MetadataRAGState, documents_sha256: str
) -> dict[str, Any]:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "dataset_id": state.dataset_id,
        "document_count": len(state.documents),
        "state_version": state.state_version,
        "schema_version": state.schema_version,
        "extractor_version": state.extractor_version,
        "adapter_version": ADAPTER_VERSION,
        "documents_sha256": documents_sha256,
        "generation_enabled": False,
        "retrieval_execution_enabled": False,
        "evaluation_execution_enabled": False,
    }


def write_state(
    state: MetadataRAGState, output_dir: Path
) -> dict[str, str]:
    """Write a Metadata-RAG state to disk and return the checksum bundle."""
    if not isinstance(state, MetadataRAGState):
        raise MetadataRAGPersistenceError(
            "write_state requires a MetadataRAGState"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    documents_sorted = sorted(
        state.documents, key=lambda d: d.document_order
    )
    doc_payloads = [_serialize_document(d) for d in documents_sorted]
    documents_bytes = _encode_jsonl(doc_payloads)
    documents_sha = _sha256_bytes(documents_bytes)

    manifest = build_manifest(state, documents_sha)
    manifest_bytes = _encode_json(manifest)

    _atomic_write_bytes(output_dir / DOCUMENTS_FILE, documents_bytes)
    _atomic_write_bytes(output_dir / MANIFEST_FILE, manifest_bytes)

    manifest_sha = _sha256_file(output_dir / MANIFEST_FILE)
    documents_file_sha = _sha256_file(output_dir / DOCUMENTS_FILE)

    checksums = {
        DOCUMENTS_FILE: documents_file_sha,
        MANIFEST_FILE: manifest_sha,
    }
    _atomic_write_bytes(
        output_dir / CHECKSUMS_FILE, _encode_json(checksums)
    )
    return {str(k): str(v) for k, v in checksums.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MetadataRAGPersistenceError(f"missing file: {path}")
    lines: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for lineno, raw in enumerate(handle, start=1):
            stripped = raw.rstrip("\n")
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise MetadataRAGPersistenceError(
                    f"{path.name}:{lineno} invalid JSON: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise MetadataRAGPersistenceError(
                    f"{path.name}:{lineno} line is not a JSON object"
                )
            lines.append(cast("dict[str, Any]", parsed))
    return lines


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MetadataRAGPersistenceError(f"missing file: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataRAGPersistenceError(
            f"{path.name} invalid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise MetadataRAGPersistenceError(
            f"{path.name} must be a JSON object"
        )
    return cast("dict[str, Any]", parsed)


def verify_checksums(input_dir: Path) -> dict[str, str]:
    """Verify on-disk checksums. Returns the verified mapping."""
    checksums_path = input_dir / CHECKSUMS_FILE
    checksums = _read_json(checksums_path)
    for name in (DOCUMENTS_FILE, MANIFEST_FILE):
        if name not in checksums:
            raise MetadataRAGPersistenceError(
                f"checksums.json missing entry for {name}"
            )
        observed = _sha256_file(input_dir / name)
        if observed != checksums[name]:
            raise MetadataRAGPersistenceError(
                f"checksum mismatch for {name}: "
                f"observed {observed}, declared {checksums[name]}"
            )
    return {str(k): str(v) for k, v in checksums.items()}


def read_state(input_dir: Path) -> MetadataRAGState:
    """Read and revalidate a persisted Metadata-RAG state."""
    verify_checksums(input_dir)
    manifest = _read_json(input_dir / MANIFEST_FILE)

    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise MetadataRAGPersistenceError(
            "manifest.artifact_type mismatch"
        )
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise MetadataRAGPersistenceError(
            "manifest.artifact_version mismatch"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MetadataRAGPersistenceError(
            "manifest.schema_version mismatch"
        )
    if manifest.get("extractor_version") != EXTRACTOR_VERSION:
        raise MetadataRAGPersistenceError(
            "manifest.extractor_version mismatch"
        )
    if manifest.get("state_version") != STATE_VERSION:
        raise MetadataRAGPersistenceError(
            "manifest.state_version mismatch"
        )

    raw_docs = _read_jsonl(input_dir / DOCUMENTS_FILE)
    if len(raw_docs) != manifest.get("document_count"):
        raise MetadataRAGPersistenceError(
            "manifest.document_count mismatch"
        )

    documents = tuple(
        MetadataRAGIndexedDocument.model_validate(row) for row in raw_docs
    )

    dataset_id = str(manifest.get("dataset_id", ""))
    if not dataset_id:
        raise MetadataRAGPersistenceError("manifest.dataset_id missing")

    return MetadataRAGState(dataset_id=dataset_id, documents=documents)
