"""Deterministic persistence for the synthetic pilot dataset (Step 78)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rag_system.pilot_dataset import (
    DATASET_ID,
    DATASET_REVISION,
    PilotCanonicalChunk,
    PilotCanonicalDocument,
    PilotCanonicalJudgment,
    PilotCanonicalQuery,
    PilotDatasetManifest,
    validate_infrastructure_pilot_dataset,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_infrastructure_pilot_dataset(
    *,
    root: Path,
    documents: tuple[PilotCanonicalDocument, ...],
    chunks: tuple[PilotCanonicalChunk, ...],
    queries: tuple[PilotCanonicalQuery, ...],
    judgments: tuple[PilotCanonicalJudgment, ...],
    overwrite: bool = False,
) -> PilotDatasetManifest:
    validate_infrastructure_pilot_dataset(
        documents=documents, chunks=chunks, queries=queries, judgments=judgments,
    )

    root = Path(root)
    names = (
        "documents.jsonl", "chunks.jsonl", "queries.jsonl",
        "judgments.jsonl", "dataset.manifest.json", "checksums.json",
    )
    if not overwrite and any((root / n).is_file() for n in names):
        raise FileExistsError("pilot dataset already exists")

    doc_text = _jsonl(tuple(sorted(documents, key=lambda d: d.doc_id)))
    chunk_text = _jsonl(
        tuple(sorted(chunks, key=lambda c: (c.doc_id, c.source_order, c.chunk_id)))
    )
    query_text = _jsonl(tuple(sorted(queries, key=lambda q: q.query_id)))
    judgment_text = _jsonl(
        tuple(sorted(judgments, key=lambda j: (j.query_id, j.item_id)))
    )

    root.mkdir(parents=True, exist_ok=True)
    _atomic_write(root / "documents.jsonl", doc_text)
    _atomic_write(root / "chunks.jsonl", chunk_text)
    _atomic_write(root / "queries.jsonl", query_text)
    _atomic_write(root / "judgments.jsonl", judgment_text)

    unique_groups = {d.content_group_id for d in documents}
    manifest = PilotDatasetManifest(
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        split="pilot",
        document_count=len(documents),
        unique_content_count=len(unique_groups),
        duplicate_content_pair_count=len(unique_groups),
        chunk_count=len(chunks),
        query_count=len(queries),
        judgment_count=len(judgments),
        documents_path="documents.jsonl",
        chunks_path="chunks.jsonl",
        queries_path="queries.jsonl",
        judgments_path="judgments.jsonl",
        documents_sha256=_sha256_bytes(doc_text.encode("utf-8")),
        chunks_sha256=_sha256_bytes(chunk_text.encode("utf-8")),
        queries_sha256=_sha256_bytes(query_text.encode("utf-8")),
        judgments_sha256=_sha256_bytes(judgment_text.encode("utf-8")),
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
    _atomic_write(root / "dataset.manifest.json", manifest_text)

    checksums = {
        "documents.jsonl": _sha256_bytes(doc_text.encode("utf-8")),
        "chunks.jsonl": _sha256_bytes(chunk_text.encode("utf-8")),
        "queries.jsonl": _sha256_bytes(query_text.encode("utf-8")),
        "judgments.jsonl": _sha256_bytes(judgment_text.encode("utf-8")),
        "dataset.manifest.json": _sha256_bytes(manifest_text.encode("utf-8")),
    }
    checksums_text = (
        json.dumps(checksums, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write(root / "checksums.json", checksums_text)

    return manifest


def read_infrastructure_pilot_dataset(
    *,
    root: Path,
    expected_dataset_id: str | None = None,
    expected_dataset_revision: str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "dataset.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    manifest = PilotDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    checksums = json.loads(
        (root / "checksums.json").read_text(encoding="utf-8")
    )
    for name in (
        "documents.jsonl", "chunks.jsonl",
        "queries.jsonl", "judgments.jsonl", "dataset.manifest.json",
    ):
        actual = _sha256_bytes((root / name).read_bytes())
        if checksums[name] != actual:
            raise RuntimeError(f"checksum mismatch: {name}")

    def _read(name: str, model_type: Any) -> tuple[Any, ...]:
        return tuple(
            model_type.model_validate_json(line)
            for line in (root / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    documents = _read(manifest.documents_path, PilotCanonicalDocument)
    chunks = _read(manifest.chunks_path, PilotCanonicalChunk)
    queries = _read(manifest.queries_path, PilotCanonicalQuery)
    judgments = _read(manifest.judgments_path, PilotCanonicalJudgment)

    if len(documents) != manifest.document_count:
        raise RuntimeError("document_count mismatch")
    if len(chunks) != manifest.chunk_count:
        raise RuntimeError("chunk_count mismatch")
    if len(queries) != manifest.query_count:
        raise RuntimeError("query_count mismatch")
    if len(judgments) != manifest.judgment_count:
        raise RuntimeError("judgment_count mismatch")

    if expected_dataset_id is not None and manifest.dataset_id != expected_dataset_id:
        raise RuntimeError("dataset_id mismatch")
    if (
        expected_dataset_revision is not None
        and manifest.dataset_revision != expected_dataset_revision
    ):
        raise RuntimeError("dataset_revision mismatch")

    validate_infrastructure_pilot_dataset(
        documents=documents, chunks=chunks, queries=queries, judgments=judgments,
    )

    return {
        "manifest": manifest,
        "documents": documents,
        "chunks": chunks,
        "queries": queries,
        "judgments": judgments,
        "checksums": checksums,
    }
