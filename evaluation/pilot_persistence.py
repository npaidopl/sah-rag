"""Deterministic persistence for Step 89 pilot evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_system.evaluation import RetrievalEvaluationCase
from rag_system.evaluation_contracts import (
    AggregateCutoffMetrics,
    QueryCutoffMetrics,
)

RETRIEVAL_CASES_FILE = "retrieval_cases.jsonl"
QUERY_METRICS_FILE = "query_metrics.jsonl"
AGGREGATE_METRICS_FILE = "aggregate_metrics.json"
MANIFEST_FILE = "evaluation.manifest.json"
CHECKSUMS_FILE = "checksums.json"

REQUIRED_FILES = (
    RETRIEVAL_CASES_FILE,
    QUERY_METRICS_FILE,
    AGGREGATE_METRICS_FILE,
    MANIFEST_FILE,
    CHECKSUMS_FILE,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_jsonl(records: Sequence[dict[str, Any]]) -> bytes:
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
    text = "\n".join(lines) + ("\n" if lines else "")
    return text.encode("utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _case_sort_key(c: dict[str, Any]) -> tuple[str]:
    return (str(c["query_id"]),)


def _query_metric_sort_key(r: dict[str, Any]) -> tuple[str, int]:
    return (str(r["query_id"]), int(r["cutoff"]))


def _aggregate_sort_key(r: dict[str, Any]) -> tuple[int]:
    return (int(r["cutoff"]),)


def write_pilot_evaluation_artifacts(
    *,
    root: Path,
    retrieval_cases: Sequence[RetrievalEvaluationCase],
    query_metrics: Sequence[QueryCutoffMetrics],
    aggregate_metrics: Sequence[AggregateCutoffMetrics],
    manifest: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write deterministic evaluation artifacts under ``root``."""
    if root.exists() and not overwrite:
        raise FileExistsError(
            "Evaluation output root already exists: " + str(root)
        )

    root.mkdir(parents=True, exist_ok=True)

    cases_payload = sorted(
        [c.model_dump() for c in retrieval_cases],
        key=_case_sort_key,
    )
    qm_payload = sorted(
        [m.model_dump() for m in query_metrics],
        key=_query_metric_sort_key,
    )
    agg_payload = sorted(
        [m.model_dump() for m in aggregate_metrics],
        key=_aggregate_sort_key,
    )

    (root / RETRIEVAL_CASES_FILE).write_bytes(_dump_jsonl(cases_payload))
    (root / QUERY_METRICS_FILE).write_bytes(_dump_jsonl(qm_payload))
    (root / AGGREGATE_METRICS_FILE).write_bytes(
        (
            json.dumps(agg_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    )
    (root / MANIFEST_FILE).write_bytes(
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    )

    checksum_targets = (
        RETRIEVAL_CASES_FILE,
        QUERY_METRICS_FILE,
        AGGREGATE_METRICS_FILE,
        MANIFEST_FILE,
    )
    checksums = {
        "artifact_type": "pilot_evaluation_checksums",
        "artifact_version": "1.0",
        "algorithm": "sha256",
        "files": {
            name: _sha256_file(root / name) for name in checksum_targets
        },
    }
    (root / CHECKSUMS_FILE).write_bytes(
        (
            json.dumps(checksums, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    )

    return {
        "retrieval_cases": len(cases_payload),
        "query_metrics": len(qm_payload),
        "aggregate_metrics": len(agg_payload),
        "root": str(root),
    }


def read_pilot_evaluation_artifacts(
    *,
    root: Path,
    expected_query_count: int = 6,
) -> dict[str, Any]:
    """Reload and validate evaluation artifacts from ``root``."""
    for name in REQUIRED_FILES:
        path = root / name
        if not path.exists():
            raise FileNotFoundError("Missing artifact: " + str(path))

    cases = _load_jsonl(root / RETRIEVAL_CASES_FILE)
    query_metrics = _load_jsonl(root / QUERY_METRICS_FILE)
    aggregate_metrics = json.loads(
        (root / AGGREGATE_METRICS_FILE).read_text(encoding="utf-8")
    )
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    checksums = json.loads((root / CHECKSUMS_FILE).read_text(encoding="utf-8"))

    query_ids = {str(c["query_id"]) for c in cases}
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
        "retrieval_cases": cases,
        "query_metrics": query_metrics,
        "aggregate_metrics": aggregate_metrics,
        "manifest": manifest,
        "checksums_valid": checksum_ok,
    }