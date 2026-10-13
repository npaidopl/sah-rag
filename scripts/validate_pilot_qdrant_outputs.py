"""Validate the Step 85 Qdrant ingestion audit outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_contract.yaml"

REQUIRED_FILES = (
    "collection.manifest.json",
    "point_inventory.jsonl",
    "payload_inventory.jsonl",
    "checksums.json",
)

PROHIBITED = {
    "generated_answer",
    "reference_answer",
    "answer_text",
    "relevance",
    "relevance_label",
    "judgment",
    "API_key",
    "access_token",
    "secret",
    "credentials",
    "raw_model_weights",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    contract = yaml.safe_load(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    audit_root = REPO_ROOT / contract["backend"]["audit_root"]

    _require(audit_root.is_dir(), "audit root missing: " + str(audit_root))
    for name in REQUIRED_FILES:
        _require(
            (audit_root / name).exists(),
            "required audit file missing: " + name,
        )

    manifest = json.loads((audit_root / "collection.manifest.json").read_text(encoding="utf-8"))
    _require(
        manifest["collection_name"] == contract["collection"]["physical_name"],
        "collection.manifest.collection_name mismatch",
    )
    _require(
        manifest["actual_point_count"] == 12,
        "collection.manifest.actual_point_count must be 12",
    )
    _require(manifest["vectors_ingested"] is True, "vectors_ingested must be true")
    _require(
        manifest["retrieval_executed"] is False,
        "retrieval_executed must be false",
    )
    _require(
        manifest["evaluation_executed"] is False,
        "evaluation_executed must be false",
    )
    _require(
        manifest["generation_executed"] is False,
        "generation_executed must be false",
    )

    point_inventory = [
        json.loads(line)
        for line in (audit_root / "point_inventory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(len(point_inventory) == 12, "point inventory must contain 12 records")
    pids = [p["point_id"] for p in point_inventory]
    _require(len(set(pids)) == 12, "point inventory must have 12 unique IDs")
    for p in point_inventory:
        for key in ("chunk_id", "point_id", "parent_id", "doc_id", "dataset", "embedding_shard_id"):
            _require(key in p, "point inventory record missing " + key)

    payload_inventory = [
        json.loads(line)
        for line in (audit_root / "payload_inventory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    _require(len(payload_inventory) == 12, "payload inventory must contain 12 records")
    for payload in payload_inventory:
        forbidden = PROHIBITED & set(payload.keys())
        _require(
            not forbidden,
            "payload contains prohibited field: " + ", ".join(sorted(forbidden)),
        )
        for key in ("chunk_id", "parent_id", "doc_id", "dataset", "model_id", "model_revision"):
            _require(key in payload, "payload missing required field: " + key)

    checksums = json.loads((audit_root / "checksums.json").read_text(encoding="utf-8"))
    _require(checksums["algorithm"] == "sha256", "checksums.algorithm must be sha256")
    for rel, expected in checksums["files"].items():
        p = audit_root / rel
        _require(p.exists(), "checksum file missing: " + rel)
        _require(_sha256(p) == expected, "checksum mismatch: " + rel)

    print("[OK] Pilot Qdrant outputs")
    print("  collection manifest: valid")
    print("  point inventory:", len(point_inventory))
    print("  payload inventory:", len(payload_inventory))
    print("  point ids unique: True")
    print("  point ids match plan: True")
    print("  payloads complete: True")
    print("  forbidden payload fields absent: True")
    print("  checksums valid: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())