"""Validate the materialized BGE-M3 pilot embeddings."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_contract.yaml"

NORM_TOLERANCE = 1.0e-4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    output_root = REPO_ROOT / contract["materialization"]["output_root"]

    _require(output_root.is_dir(), "output root missing: " + str(output_root))

    required = contract["materialization"]["required_files"]
    for name in required:
        p = output_root / name
        _require(p.exists(), "required file missing: " + name)

    shard_dir = output_root / "shards" / "shard-00000"
    header = json.loads((shard_dir / "header.json").read_text(encoding="utf-8"))
    _require(
        header["dataset"] == contract["source"]["dataset_id"],
        "header.dataset mismatch",
    )
    _require(
        header["model_id"] == contract["model"]["model_id"],
        "header.model_id mismatch",
    )
    _require(
        header["model_revision"] == contract["model"]["model_revision"],
        "header.model_revision mismatch",
    )
    _require(
        int(header["dense_dimension"]) == 1024,
        "header.dense_dimension must be 1024",
    )
    _require(
        bool(header["dense_normalized"]) is True,
        "header.dense_normalized must be true",
    )

    records_path = shard_dir / "records.jsonl"
    records = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    expected = int(contract["source"]["expected_child_chunk_count"])
    _require(len(records) == expected, "record count mismatch: " + str(len(records)))

    seen_ids = set()
    for rec in records:
        cid = rec["chunk_id"]
        _require(cid not in seen_ids, "duplicate chunk_id: " + cid)
        seen_ids.add(cid)
        _require("parent_id" in rec and rec["parent_id"], "missing parent_id")
        _require("doc_id" in rec and rec["doc_id"], "missing doc_id")
        _require(
            rec["dataset"] == contract["source"]["dataset_id"],
            "record.dataset mismatch",
        )
        _require(
            rec["model_id"] == contract["model"]["model_id"],
            "record.model_id mismatch",
        )

        dense = rec.get("dense_vector")
        _require(
            isinstance(dense, list) and len(dense) == 1024,
            "dense_vector must be 1024 floats",
        )
        for x in dense:
            _require(
                isinstance(x, (int, float)) and math.isfinite(float(x)),
                "dense_value must be finite",
            )
        norm = math.sqrt(sum(float(x) * float(x) for x in dense))
        _require(
            abs(norm - 1.0) <= NORM_TOLERANCE,
            "dense_vector not L2-normalized (norm=" + str(norm) + ")",
        )

        sparse = rec.get("sparse_vector")
        _require(isinstance(sparse, dict), "sparse_vector must be a mapping")
        indices = sparse.get("indices")
        values = sparse.get("values")
        _require(
            isinstance(indices, list) and isinstance(values, list),
            "sparse indices/values must be lists",
        )
        _require(
            len(indices) == len(values),
            "sparse indices/values length mismatch",
        )
        _require(
            all(isinstance(i, int) and i >= 0 for i in indices),
            "sparse indices must be non-negative ints",
        )
        _require(indices == sorted(indices), "sparse indices must be ascending")
        _require(len(set(indices)) == len(indices), "sparse indices must be unique")
        _require(
            all(
                isinstance(v, (int, float))
                and math.isfinite(float(v))
                and float(v) >= 0.0
                for v in values
            ),
            "sparse values must be finite non-negative",
        )

    catalog_path = output_root / "embedding.catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    _require(
        catalog["dataset_id"] == contract["source"]["dataset_id"],
        "catalog.dataset_id mismatch",
    )
    _require(
        catalog["model_id"] == contract["model"]["model_id"],
        "catalog.model_id mismatch",
    )
    _require(
        catalog["model_revision"] == contract["model"]["model_revision"],
        "catalog.model_revision mismatch",
    )
    _require(
        catalog["record_count"] == expected,
        "catalog.record_count mismatch",
    )
    _require(catalog["shard_count"] == 1, "catalog.shard_count must be 1")

    manifest_path = output_root / "embedding.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest["record_count"] == expected,
        "manifest.record_count mismatch",
    )
    _require(
        manifest["determinism_status"]
        in ("BYTE_IDENTICAL", "NUMERICALLY_STABLE_NOT_BYTE_IDENTICAL"),
        "manifest.determinism_status invalid",
    )
    _require(
        manifest["round_trip_valid"] is True,
        "manifest.round_trip_valid must be true",
    )

    checksums_path = output_root / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    _require(
        checksums["algorithm"] == "sha256",
        "checksums.algorithm must be sha256",
    )
    for rel, expected_sha in checksums["files"].items():
        p = output_root / rel
        _require(p.exists(), "checksums references missing file: " + rel)
        _require(_sha256(p) == expected_sha, "checksum mismatch: " + rel)

    print("[OK] Pilot BGE-M3 embedding outputs")
    print("  records:", len(records))
    print("  dense dimension: 1024")
    print("  sparse vectors:", sum(1 for r in records if r.get("sparse_vector")))
    print("  lineage preserved: True")
    print("  round trip: True")
    print("  checksums valid: True")
    print("  determinism:", manifest["determinism_status"])
    print("  shards:", catalog["shard_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())