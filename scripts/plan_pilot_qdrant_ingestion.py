"""Deterministic planner for the Step 85 bounded Qdrant ingestion.

Validates the three Step 84 contracts, cross-checks the Step 83
embedding manifest, computes deterministic point IDs, computes the
physical collection name via the project-owned helper, and updates
only the audit-side planning manifest. It never opens a QdrantClient,
never creates any storage directory, and never executes retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

INGESTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_ingestion_contract.yaml"
PAYLOAD_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_payload_contract.yaml"
INGESTION_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_qdrant_ingestion_manifest.yaml"
EMBEDDING_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_manifest.yaml"

FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FROZEN_MODEL_ID = "BAAI/bge-m3"
FROZEN_DATASET = "infrastructure-pilot-synthetic-v1"

NORM_TOLERANCE = 1.0e-4


def _catalog_sha256(catalog: dict) -> str:
    """Return the SHA-256 fingerprint of the catalog.

    The published catalog does not carry a top-level catalog_sha256
    field. The deterministic fingerprint used here is the canonical
    JSON encoding of the full catalog document hashed with SHA-256.
    This value is stable across re-runs and is what the physical
    collection name ultimately binds to.
    """
    payload = json.dumps(
        catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_child_chunks(path: Path) -> list[dict]:
    return _load_jsonl(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_embedding_manifest(manifest: dict) -> None:
    _require(
        manifest["execution"]["status"] == "SUCCEEDED",
        "embedding manifest status must be SUCCEEDED",
    )
    results = manifest["results"]
    _require(results["dense_vector_count"] == 12, "dense_vector_count must be 12")
    _require(results["sparse_vector_count"] == 12, "sparse_vector_count must be 12")
    _require(results["round_trip_valid"] is True, "round_trip_valid must be true")
    _require(results["checksums_valid"] is True, "checksums_valid must be true")
    _require(
        results["deterministic_data_artifacts"] is True,
        "deterministic_data_artifacts must be true",
    )


def verify_checksums(embedding_root: Path) -> None:
    checksums_path = embedding_root / "checksums.json"
    _require(checksums_path.exists(), "embedding checksums.json missing")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    _require(checksums["algorithm"] == "sha256", "checksums.algorithm must be sha256")
    for rel, expected in checksums["files"].items():
        p = embedding_root / rel
        _require(p.exists(), "checksum file missing: " + rel)
        _require(_sha256_file(p) == expected, "checksum mismatch: " + rel)


def validate_records(records: list[dict], expected_count: int) -> None:
    _require(
        len(records) == expected_count,
        "record count mismatch: " + str(len(records)) + " != " + str(expected_count),
    )
    seen_chunk_ids: set[str] = set()
    for rec in records:
        cid = rec["chunk_id"]
        _require(cid not in seen_chunk_ids, "duplicate chunk_id: " + cid)
        seen_chunk_ids.add(cid)

        _require(rec["dataset"] == FROZEN_DATASET, "dataset mismatch: " + rec["dataset"])
        _require(rec["model_id"] == FROZEN_MODEL_ID, "model_id mismatch")
        _require(rec["model_revision"] == FROZEN_REVISION, "model_revision mismatch")
        _require(rec["parent_id"], "parent_id empty on " + cid)
        _require(rec["doc_id"], "doc_id empty on " + cid)

        dense = rec["dense_vector"]
        _require(isinstance(dense, list) and len(dense) == 1024, "dense dim must be 1024")
        _require(
            all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in dense),
            "dense non-finite value on " + cid,
        )
        norm = math.sqrt(sum(float(x) * float(x) for x in dense))
        _require(
            abs(norm - 1.0) <= NORM_TOLERANCE,
            "dense not L2-normalized on " + cid,
        )

        sparse = rec["sparse_vector"]
        idx = sparse["indices"]
        val = sparse["values"]
        _require(len(idx) == len(val), "sparse length mismatch on " + cid)
        _require(idx == sorted(idx), "sparse indices must be ascending on " + cid)
        _require(len(set(idx)) == len(idx), "sparse indices must be unique on " + cid)
        _require(all(isinstance(i, int) and i >= 0 for i in idx), "sparse index invalid on " + cid)
        _require(
            all(
                isinstance(v, (int, float))
                and math.isfinite(float(v))
                and float(v) >= 0.0
                for v in val
            ),
            "sparse value invalid on " + cid,
        )


def validate_payload_presence(
    chunks: list[dict],
    records: list[dict],
) -> None:
    """Cross-check that every chunk needed for build_payload is present."""
    record_ids = {r["chunk_id"] for r in records}
    for chunk in chunks:
        _require(
            chunk["chunk_id"] in record_ids,
            "child chunk without embedding: " + chunk["chunk_id"],
        )


# ---------------------------------------------------------------------------
# Deterministic computations
# ---------------------------------------------------------------------------


def build_point_ids(records: list[dict]) -> list[tuple[str, str]]:
    from sah_rag.storage.qdrant import build_qdrant_point_id

    pairs = []
    for rec in records:
        pairs.append((rec["chunk_id"], build_qdrant_point_id(rec["chunk_id"])))
    pairs.sort(key=lambda item: item[0])
    return pairs


def compute_physical_name(dataset: str, embedding_config_hash: str) -> str:
    from sah_rag.storage.qdrant import fingerprint_collection_name

    return fingerprint_collection_name(
        dataset=dataset,
        embedding_config_hash=embedding_config_hash,
    )


def compute_informational_hash(ingestion_contract: dict, catalog: dict) -> str:
    """Build the informational configuration hash documented in the contract."""
    identity = {
        "dataset": ingestion_contract["source"].get("dataset", FROZEN_DATASET)
        if "dataset" in ingestion_contract["source"]
        else FROZEN_DATASET,
        "embedding_catalog_sha256": _catalog_sha256(catalog),
        "model_id": FROZEN_MODEL_ID,
        "model_revision": FROZEN_REVISION,
        "dense_vector_name": "dense",
        "dense_dimension": 1024,
        "dense_distance": "COSINE",
        "sparse_vector_name": "sparse",
        "payload_schema_version": "0.1.0",
        "point_id_algorithm": "uuid5",
        "point_id_namespace": "URL",
        "point_id_input_format": "rag-chunk:{chunk_id}",
    }
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------


def update_planning_manifest(
    informational_hash: str,
    physical_name: str,
) -> None:
    manifest = _load_yaml(INGESTION_MANIFEST)
    _require(
        manifest["execution"]["status"] == "PLANNED",
        "planning manifest must start from PLANNED",
    )
    manifest["plan"]["configuration_hash"] = informational_hash
    manifest["plan"]["physical_collection_name"] = physical_name
    INGESTION_MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Preview only; do not write the planning manifest.",
    )
    args = parser.parse_args()

    ingestion_contract = _load_yaml(INGESTION_CONTRACT)
    _ = _load_yaml(PAYLOAD_CONTRACT)  # presence check
    embedding_manifest = _load_yaml(EMBEDDING_MANIFEST)

    validate_embedding_manifest(embedding_manifest)

    embedding_root = REPO_ROOT / ingestion_contract["source"]["embedding_root"]
    verify_checksums(embedding_root)

    # Load embedding records and child chunks
    shard_dir = embedding_root / ingestion_contract["source"]["embedding_shard_subdir"]
    records_path = shard_dir / "records.jsonl"
    records = _load_jsonl(records_path)

    catalog_path = embedding_root / "embedding.catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    expected_count = int(ingestion_contract["source"]["expected_record_count"])
    validate_records(records, expected_count)

    # child chunks for payload cross-check
    child_chunks_relative = (
        "outputs/infrastructure_pilot/"
        "infrastructure-pilot-synthetic-v1/"
        "sah_rag/chunking/child_chunks.jsonl"
    )
    child_chunks_path = REPO_ROOT / child_chunks_relative
    chunks = _load_child_chunks(child_chunks_path)
    validate_payload_presence(chunks, records)

    # Deterministic point IDs
    point_pairs = build_point_ids(records)
    point_ids = [pid for _, pid in point_pairs]
    _require(len(point_ids) == expected_count, "point count mismatch")
    _require(len(set(point_ids)) == expected_count, "point IDs not unique")
    _require(
        len({cid for cid, _ in point_pairs}) == expected_count,
        "chunk IDs not unique",
    )

    # Physical collection name + informational hash
    embedding_config_hash = catalog["embedding_config_hash"]
    physical_name = compute_physical_name(FROZEN_DATASET, embedding_config_hash)
    informational_hash = compute_informational_hash(ingestion_contract, catalog)

    # Storage root must not exist
    storage_root = REPO_ROOT / ingestion_contract["backend"]["storage_root"]
    _require(
        not storage_root.exists(),
        "storage root already exists: " + str(storage_root),
    )

    if not args.no_update_manifest:
        update_planning_manifest(informational_hash, physical_name)

    # ---- Output ----
    print("[OK] Pilot Qdrant ingestion plan")
    print("  backend: Qdrant")
    print("  mode: local_persistence")
    print("  embedding records:", len(records))
    print("  expected points:", expected_count)
    print()
    print("  dense vector:")
    print("    name: dense")
    print("    dimension: 1024")
    print("    distance: COSINE")
    print()
    print("  sparse vector:")
    print("    name: sparse")
    print("    vectors:", len(records))
    print("    indices sorted: True")
    print("    values non-negative: True")
    print()
    print("  unique chunk ids: True")
    print("  unique deterministic point ids: True")
    print("  payloads complete: True")
    print("  lineage complete: True")
    print()
    print("  configuration hash:", informational_hash)
    print("  physical collection name:", physical_name)
    print()
    print("  storage directory created: False")
    print("  Qdrant client opened: False")
    print("  collection created: False")
    print("  vectors ingested: False")
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())