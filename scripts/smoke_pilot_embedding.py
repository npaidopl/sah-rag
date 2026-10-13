"""Fast smoke check for the materialized pilot embeddings.

Reads the shard and sidecar files, verifies:
- record count and lineage
- dense dimension and finite values
- sparse index ordering and non-negative values
- catalog/manifest/checksums coherence

Does NOT load the model and does NOT re-encode.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml"


def _fail(message: str) -> None:
    raise SystemExit("[FAIL] " + message)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    output_root = REPO_ROOT / contract["materialization"]["output_root"]
    if not output_root.is_dir():
        _fail("output root missing: " + str(output_root))

    shard_dir = output_root / "shards" / "shard-00000"
    records_path = shard_dir / "records.jsonl"
    catalog_path = output_root / "embedding.catalog.json"
    manifest_path = output_root / "embedding.manifest.json"
    checksums_path = output_root / "checksums.json"

    for p in (records_path, catalog_path, manifest_path, checksums_path):
        if not p.exists():
            _fail("required file missing: " + str(p))

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = int(contract["source"]["expected_child_chunk_count"])
    if len(records) != expected:
        _fail("record count " + str(len(records)) + " != " + str(expected))

    seen: set[str] = set()
    for r in records:
        if r["chunk_id"] in seen:
            _fail("duplicate chunk_id: " + r["chunk_id"])
        seen.add(r["chunk_id"])
        if not r.get("parent_id") or not r.get("doc_id"):
            _fail("lineage missing on " + r["chunk_id"])
        dense = r.get("dense_vector") or []
        if len(dense) != 1024:
            _fail("dense dimension wrong on " + r["chunk_id"])
        if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in dense):
            _fail("non-finite dense value on " + r["chunk_id"])
        norm = math.sqrt(sum(float(x) * float(x) for x in dense))
        if abs(norm - 1.0) > 1.0e-4:
            _fail("dense not L2-normalized on " + r["chunk_id"] + " norm=" + str(norm))
        sparse = r.get("sparse_vector") or {}
        idx = sparse.get("indices") or []
        val = sparse.get("values") or []
        if len(idx) != len(val):
            _fail("sparse length mismatch on " + r["chunk_id"])
        if idx != sorted(idx):
            _fail("sparse indices not ascending on " + r["chunk_id"])
        if len(set(idx)) != len(idx):
            _fail("sparse indices not unique on " + r["chunk_id"])
        if not all(isinstance(i, int) and i >= 0 for i in idx):
            _fail("sparse index invalid on " + r["chunk_id"])
        values_ok = all(
            isinstance(v, (int, float))
            and math.isfinite(float(v))
            and float(v) >= 0.0
            for v in val
        )
        if not values_ok:
            _fail("sparse value invalid on " + r["chunk_id"])

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog["record_count"] != expected or catalog["shard_count"] != 1:
        _fail("catalog coherence check failed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["record_count"] != expected:
        _fail("manifest record_count mismatch")
    if manifest["determinism_status"] not in (
        "BYTE_IDENTICAL",
        "NUMERICALLY_STABLE_NOT_BYTE_IDENTICAL",
    ):
        _fail("determinism status invalid: " + str(manifest["determinism_status"]))

    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for rel, expected_sha in checksums["files"].items():
        p = output_root / rel
        if not p.exists():
            _fail("checksums references missing file: " + rel)
        if _sha256(p) != expected_sha:
            _fail("checksum mismatch: " + rel)

    print("[OK] Pilot BGE-M3 embedding smoke")
    print("  records:", len(records))
    print("  shards: 1")
    print("  dense dimension: 1024")
    print("  determinism:", manifest["determinism_status"])
    print("  round trip:", manifest["round_trip_valid"])
    print("  checksums valid: True")
    print("  model download performed: False")
    print("  network access performed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())