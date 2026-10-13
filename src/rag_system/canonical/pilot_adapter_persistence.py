"""Deterministic persistence for the pilot canonical adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rag_system.canonical.models import CanonicalDocument, CanonicalExample
from rag_system.canonical.pilot_adapter import (
    PilotAdapterManifest,
    PilotAdapterValidation,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _jsonl(models: tuple[Any, ...]) -> str:
    return "".join(
        json.dumps(
            m.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for m in models
    )


def write_pilot_canonical_adapter(
    *,
    target_root: Path,
    documents: tuple[CanonicalDocument, ...],
    examples: tuple[CanonicalExample, ...],
    validation: PilotAdapterValidation,
    mapping: dict[str, Any],
    overwrite: bool = False,
) -> PilotAdapterManifest:
    target_root = Path(target_root)
    names = (
        "documents.jsonl",
        "examples.jsonl",
        "validation.json",
        "manifest.json",
        "adapter.mapping.json",
        "checksums.json",
    )
    if not overwrite and any((target_root / n).is_file() for n in names):
        raise FileExistsError("pilot canonical adapter target already exists")

    docs_ordered = tuple(sorted(documents, key=lambda d: d.doc_id))
    ex_ordered = tuple(sorted(examples, key=lambda e: e.query.query_id))

    docs_text = _jsonl(docs_ordered)
    ex_text = _jsonl(ex_ordered)
    val_text = (
        json.dumps(
            validation.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    map_text = (
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    target_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(target_root / "documents.jsonl", docs_text)
    _atomic_write(target_root / "examples.jsonl", ex_text)
    _atomic_write(target_root / "validation.json", val_text)
    _atomic_write(target_root / "adapter.mapping.json", map_text)

    docs_sha = _sha256_bytes(docs_text.encode("utf-8"))
    ex_sha = _sha256_bytes(ex_text.encode("utf-8"))
    val_sha = _sha256_bytes(val_text.encode("utf-8"))
    map_sha = _sha256_bytes(map_text.encode("utf-8"))

    source_manifest_sha = str(mapping["source_manifest_sha256"])

    manifest = PilotAdapterManifest(
        adapter_id=str(mapping["adapter_id"]),
        adapter_version=str(mapping["adapter_version"]),
        source_dataset_id=str(mapping["source_dataset_id"]),
        source_dataset_revision=str(mapping["source_dataset_revision"]),
        source_manifest_sha256=source_manifest_sha,
        target_dataset_id=str(mapping["source_dataset_id"]),
        document_count=len(docs_ordered),
        example_count=len(ex_ordered),
        document_mapping_count=len(mapping.get("document_mappings") or []),
        example_mapping_count=len(mapping.get("example_mappings") or []),
        documents_sha256=docs_sha,
        examples_sha256=ex_sha,
        validation_sha256=val_sha,
        mapping_sha256=map_sha,
    )

    manifest_text = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_write(target_root / "manifest.json", manifest_text)

    checksums = {
        "documents.jsonl": docs_sha,
        "examples.jsonl": ex_sha,
        "validation.json": val_sha,
        "adapter.mapping.json": map_sha,
        "manifest.json": _sha256_bytes(manifest_text.encode("utf-8")),
    }
    checksums_text = (
        json.dumps(checksums, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write(target_root / "checksums.json", checksums_text)

    return manifest


def read_pilot_canonical_adapter(
    *,
    target_root: Path,
    expected_adapter_id: str | None = None,
) -> dict[str, Any]:
    target_root = Path(target_root)
    manifest_path = target_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    manifest = PilotAdapterManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    checksums = json.loads(
        (target_root / "checksums.json").read_text(encoding="utf-8")
    )
    for name in (
        "documents.jsonl",
        "examples.jsonl",
        "validation.json",
        "adapter.mapping.json",
        "manifest.json",
    ):
        actual = _sha256_bytes((target_root / name).read_bytes())
        if checksums[name] != actual:
            raise RuntimeError(f"checksum mismatch: {name}")

    def _read(name: str, model_type: Any) -> tuple[Any, ...]:
        return tuple(
            model_type.model_validate_json(line)
            for line in (target_root / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    documents = _read("documents.jsonl", CanonicalDocument)
    examples = _read("examples.jsonl", CanonicalExample)
    validation = PilotAdapterValidation.model_validate_json(
        (target_root / "validation.json").read_text(encoding="utf-8")
    )
    mapping = json.loads(
        (target_root / "adapter.mapping.json").read_text(encoding="utf-8")
    )

    if len(documents) != manifest.document_count:
        raise RuntimeError("document count mismatch")
    if len(examples) != manifest.example_count:
        raise RuntimeError("example count mismatch")

    if expected_adapter_id is not None and manifest.adapter_id != expected_adapter_id:
        raise RuntimeError("adapter_id mismatch")

    return {
        "manifest": manifest,
        "documents": documents,
        "examples": examples,
        "validation": validation,
        "mapping": mapping,
        "checksums": checksums,
    }
