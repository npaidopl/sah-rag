"""Deterministic persistence for Step 87 pilot retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_system.retrieval.document_resolution import RetrievedDocument

QUERY_EMBEDDINGS_FILE = "query_embeddings.jsonl"
RANKED_CHILDREN_FILE = "ranked_children.jsonl"
RANKED_PARENTS_FILE = "ranked_parents.jsonl"
RANKED_DOCUMENTS_FILE = "ranked_documents.jsonl"
MANIFEST_FILE = "retrieval.manifest.json"
CHECKSUMS_FILE = "checksums.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_jsonl(records: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        lines.append(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sorted_query_embeddings(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: str(r["query_id"]))


def _sorted_ranked_children(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            str(r["query_id"]),
            str(r["retrieval_method"]),
            int(r["rank"]),
            str(r["chunk_id"]),
        ),
    )


def _sorted_ranked_parents(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            str(r["query_id"]),
            int(r["rank"]),
            str(r["parent_id"]),
        ),
    )


def _sorted_ranked_documents(
    records: Sequence[RetrievedDocument],
) -> list[dict[str, Any]]:
    dumped: list[dict[str, Any]] = [r.model_dump() for r in records]
    return sorted(
        dumped,
        key=lambda r: (
            str(r["query_id"]),
            int(r["rank"]),
            str(r["doc_id"]),
        ),
    )


def write_pilot_retrieval_artifacts(
    *,
    root: Path,
    query_embeddings: Sequence[dict[str, Any]],
    ranked_children: Sequence[dict[str, Any]],
    ranked_parents: Sequence[dict[str, Any]],
    ranked_documents: Sequence[RetrievedDocument],
    manifest: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write deterministic retrieval artifacts under ``root``."""
    if root.exists() and not overwrite:
        raise FileExistsError(
            "Retrieval output root already exists: " + str(root)
        )

    root.mkdir(parents=True, exist_ok=True)

    qe_sorted = _sorted_query_embeddings(query_embeddings)
    rc_sorted = _sorted_ranked_children(ranked_children)
    rp_sorted = _sorted_ranked_parents(ranked_parents)
    rd_sorted = _sorted_ranked_documents(ranked_documents)

    (root / QUERY_EMBEDDINGS_FILE).write_text(
        _dump_jsonl(qe_sorted), encoding="utf-8", newline="\n"
    )
    (root / RANKED_CHILDREN_FILE).write_text(
        _dump_jsonl(rc_sorted), encoding="utf-8", newline="\n"
    )
    (root / RANKED_PARENTS_FILE).write_text(
        _dump_jsonl(rp_sorted), encoding="utf-8", newline="\n"
    )
    (root / RANKED_DOCUMENTS_FILE).write_text(
        _dump_jsonl(rd_sorted), encoding="utf-8", newline="\n"
    )

    (root / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    checksum_targets = (
        QUERY_EMBEDDINGS_FILE,
        RANKED_CHILDREN_FILE,
        RANKED_PARENTS_FILE,
        RANKED_DOCUMENTS_FILE,
        MANIFEST_FILE,
    )
    checksums = {
        "artifact_type": "pilot_retrieval_checksums",
        "artifact_version": "1.0",
        "algorithm": "sha256",
        "files": {
            name: _sha256_file(root / name) for name in checksum_targets
        },
    }
    (root / CHECKSUMS_FILE).write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return {
        "query_embeddings": len(qe_sorted),
        "ranked_children": len(rc_sorted),
        "ranked_parents": len(rp_sorted),
        "ranked_documents": len(rd_sorted),
        "root": str(root),
    }


def read_pilot_retrieval_artifacts(
    *,
    root: Path,
    expected_query_count: int = 6,
) -> dict[str, Any]:
    """Reload and validate retrieval artifacts from ``root``."""
    required = (
        QUERY_EMBEDDINGS_FILE,
        RANKED_CHILDREN_FILE,
        RANKED_PARENTS_FILE,
        RANKED_DOCUMENTS_FILE,
        MANIFEST_FILE,
        CHECKSUMS_FILE,
    )
    for name in required:
        path = root / name
        if not path.exists():
            raise FileNotFoundError("Missing artifact: " + str(path))

    query_embeddings = _load_jsonl(root / QUERY_EMBEDDINGS_FILE)
    ranked_children = _load_jsonl(root / RANKED_CHILDREN_FILE)
    ranked_parents = _load_jsonl(root / RANKED_PARENTS_FILE)
    ranked_documents = _load_jsonl(root / RANKED_DOCUMENTS_FILE)
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    checksums = json.loads((root / CHECKSUMS_FILE).read_text(encoding="utf-8"))

    query_ids = {str(q["query_id"]) for q in query_embeddings}
    if len(query_ids) != expected_query_count:
        raise ValueError(
            "Expected "
            + str(expected_query_count)
            + " unique query ids, found "
            + str(len(query_ids))
        )

    checksum_ok = True
    files_map: dict[str, str] = checksums["files"]
    for name, expected in files_map.items():
        if _sha256_file(root / name) != expected:
            checksum_ok = False
            break

    return {
        "query_embeddings": query_embeddings,
        "ranked_children": ranked_children,
        "ranked_parents": ranked_parents,
        "ranked_documents": ranked_documents,
        "manifest": manifest,
        "checksums_valid": checksum_ok,
    }