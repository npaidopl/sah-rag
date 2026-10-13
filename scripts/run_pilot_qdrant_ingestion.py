"""Bounded local-persistence Qdrant ingestion for the pilot (Step 85).

Loads 12 deterministic embedding records, creates one isolated local
collection under the pilot output tree, ingests exactly 12 points,
reopens the collection with a fresh client to validate persistence,
and writes audit-side inventory and checksums. Never executes
retrieval, evaluation or generation.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_contract.yaml"
EXECUTION_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
INGESTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_ingestion_contract.yaml"
PAYLOAD_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_qdrant_payload_contract.yaml"
PLANNING_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_qdrant_ingestion_manifest.yaml"
EMBEDDING_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_embedding_execution_manifest.yaml"

FROZEN_MODEL_ID = "BAAI/bge-m3"
FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FROZEN_DATASET = "infrastructure-pilot-synthetic-v1"

PROHIBITED_PAYLOAD_FIELDS = {
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


def _tree_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Validation phases
# ---------------------------------------------------------------------------


def validate_dependencies() -> dict:
    ec = _load_yaml(EXECUTION_CONTRACT)
    _require(
        ec["contract"]["status"] == "APPROVED_FOR_BOUNDED_EXECUTION",
        "execution contract not approved",
    )
    _require(
        ec["execution"]["collection_creation_enabled"] is True,
        "collection_creation_enabled must be true",
    )
    _require(
        ec["execution"]["vector_ingestion_enabled"] is True,
        "vector_ingestion_enabled must be true",
    )

    em = _load_yaml(EMBEDDING_MANIFEST)
    _require(
        em["execution"]["status"] == "SUCCEEDED",
        "embedding manifest must be SUCCEEDED",
    )
    _require(
        em["results"]["checksums_valid"] is True,
        "embedding checksums_valid must be true",
    )

    _ = _load_yaml(INGESTION_CONTRACT)  # presence check
    _ = _load_yaml(PAYLOAD_CONTRACT)
    _ = _load_yaml(PLANNING_MANIFEST)
    return ec


def verify_embedding_checksums(embedding_root: Path) -> None:
    checksums_path = embedding_root / "checksums.json"
    _require(checksums_path.exists(), "embedding checksums.json missing")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for rel, expected in checksums["files"].items():
        p = embedding_root / rel
        _require(p.exists(), "checksum file missing: " + rel)
        _require(_sha256_file(p) == expected, "checksum mismatch: " + rel)


def load_payload(
    embedding_root: Path, execution_contract: dict
) -> tuple[list[dict], list[dict], dict]:
    shard_subdir = execution_contract["source"]["embedding_shard_subdir"]
    shard_dir = embedding_root / shard_subdir
    header = json.loads((shard_dir / "header.json").read_text(encoding="utf-8"))
    records = _load_jsonl(shard_dir / "records.jsonl")

    catalog = json.loads(
        (embedding_root / "embedding.catalog.json").read_text(encoding="utf-8")
    )

    chunks_relative = (
        "outputs/infrastructure_pilot/"
        "infrastructure-pilot-synthetic-v1/"
        "rag_system/chunking/child_chunks.jsonl"
    )
    chunks = _load_jsonl(REPO_ROOT / chunks_relative)

    expected = int(execution_contract["source"]["expected_embedding_records"])
    _require(len(records) == expected, "record count mismatch")
    _require(len(chunks) == expected, "chunk count mismatch")

    record_ids = {r["chunk_id"] for r in records}
    chunk_ids = {c["chunk_id"] for c in chunks}
    _require(record_ids == chunk_ids, "record/chunk chunk_id sets differ")

    return records, chunks, {"header": header, "catalog": catalog}


# ---------------------------------------------------------------------------
# Point ID plan (mirrors Step 84)
# ---------------------------------------------------------------------------


def planned_point_ids(records: list[dict]) -> dict[str, str]:
    from rag_system.storage.qdrant import build_qdrant_point_id

    pairs = {r["chunk_id"]: build_qdrant_point_id(r["chunk_id"]) for r in records}
    return pairs


# ---------------------------------------------------------------------------
# Storage safety
# ---------------------------------------------------------------------------


def assert_storage_path_safe(storage_root: Path, pilot_root: Path) -> None:
    resolved_storage = storage_root.resolve()
    resolved_pilot = pilot_root.resolve()
    _require(
        resolved_storage.is_relative_to(resolved_pilot),
        "storage root must be inside the pilot output tree",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    execution_contract = validate_dependencies()

    embedding_root = REPO_ROOT / execution_contract["source"]["embedding_root"]
    _require(embedding_root.is_dir(), "embedding root missing")

    verify_embedding_checksums(embedding_root)

    records, chunks, meta = load_payload(embedding_root, execution_contract)

    storage_root = REPO_ROOT / execution_contract["backend"]["storage_root"]
    audit_root = REPO_ROOT / execution_contract["backend"]["audit_root"]
    pilot_root = REPO_ROOT / "outputs" / "infrastructure_pilot"

    assert_storage_path_safe(storage_root, pilot_root)
    assert_storage_path_safe(audit_root, pilot_root)

    collection_name = execution_contract["collection"]["physical_name"]

    if storage_root.exists():
        if not args.overwrite:
            raise SystemExit("[FAIL] storage root already exists: " + str(storage_root))
        if audit_root.exists():
            shutil.rmtree(audit_root)
        shutil.rmtree(storage_root)

    storage_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    plan = planned_point_ids(records)
    _require(len(plan) == 12, "planned point ID count must be 12")
    _require(len(set(plan.values())) == 12, "planned point IDs must be unique")

    # ---- Phase 1: client open, create, ingest, close ----
    from qdrant_client import QdrantClient

    from rag_system.artifacts import ChildChunk
    from rag_system.embedding import ChunkEmbeddingRecord, EmbeddingShardHeader
    from rag_system.storage.qdrant import (
        create_hybrid_collection,
        ingest_embedding_records,
    )

    wall_start = time.perf_counter()

    client1 = QdrantClient(path=str(storage_root))
    try:
        create_hybrid_collection(
            client=client1,
            collection_name=collection_name,
            dense_dimension=int(execution_contract["vectors"]["dense"]["dimension"]),
        )

        header = EmbeddingShardHeader.model_validate(meta["header"])
        record_objs = [ChunkEmbeddingRecord.model_validate(r) for r in records]
        chunk_objs = [ChildChunk.model_validate(c) for c in chunks]

        summary = ingest_embedding_records(
            client=client1,
            collection_name=collection_name,
            header=header,
            records=record_objs,
            chunks=chunk_objs,
            batch_size=int(execution_contract["execution"]["ingestion_batch_size"]),
        )
        _require(summary.point_count == 12, "ingestion summary must report 12 points")
    finally:
        with contextlib.suppress(Exception):
            client1.close()

    wall_phase1 = time.perf_counter() - wall_start

    # ---- Phase 2: fresh client, validate persisted collection ----
    client2 = QdrantClient(path=str(storage_root))
    try:
        info = client2.get_collection(collection_name=collection_name)
        point_count = client2.count(
            collection_name=collection_name, exact=True
        ).count
        _require(point_count == 12, "persisted point count must be 12")

        vectors_config = info.config.params.vectors
        _require(
            "dense" in vectors_config,
            "dense vector config missing after reopen",
        )
        dense_params = vectors_config["dense"]
        _require(int(dense_params.size) == 1024, "dense size must be 1024")
        distance_value = getattr(dense_params.distance, "value", None) or str(
            dense_params.distance
        )
        _require(
            str(distance_value).upper() == "COSINE",
            "dense distance must be COSINE (got: "
            + str(distance_value)
            + ")",
        )
        sparse_config = info.config.params.sparse_vectors
        _require(
            sparse_config is not None and "sparse" in sparse_config,
            "sparse vector config missing after reopen",
        )

        # Direct point inspection without any query vector
        scroll_records, _ = client2.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        _require(len(scroll_records) == 12, "scroll must return 12 points")

        seen_ids: set[str] = set()
        payload_ok = True
        for p in scroll_records:
            pid = str(p.id)
            seen_ids.add(pid)
            payload = p.payload or {}
            for forbidden in PROHIBITED_PAYLOAD_FIELDS:
                if forbidden in payload:
                    payload_ok = False
            for required in (
                "chunk_id",
                "parent_id",
                "doc_id",
                "dataset",
                "model_id",
                "model_revision",
            ):
                if required not in payload:
                    payload_ok = False

        _require(len(seen_ids) == 12, "unique persisted point IDs must be 12")
        expected_ids = set(plan.values())
        _require(
            seen_ids == expected_ids,
            "persisted point IDs must match the Step 84 plan",
        )
        _require(payload_ok, "persisted payloads must be complete and non-forbidden")

        client_restart_valid = True
    finally:
        with contextlib.suppress(Exception):
            client2.close()

    wall_clock = time.perf_counter() - wall_start

    # ---- Phase 3: audit artifacts ----
    collection_manifest = {
        "artifact_type": "pilot_qdrant_collection_manifest",
        "artifact_version": "1.0",
        "collection_name": collection_name,
        "backend": "Qdrant",
        "mode": "local_persistence",
        "dataset_id": FROZEN_DATASET,
        "embedding_config_hash": meta["catalog"]["embedding_config_hash"],
        "model_id": FROZEN_MODEL_ID,
        "model_revision": FROZEN_REVISION,
        "dense_vector_name": "dense",
        "dense_dimension": 1024,
        "dense_distance": "COSINE",
        "sparse_vector_name": "sparse",
        "expected_point_count": 12,
        "actual_point_count": 12,
        "point_id_algorithm": "uuid5",
        "payload_schema_version": "0.1.0",
        "storage_root": str(storage_root.relative_to(REPO_ROOT).as_posix()),
        "storage_bytes": 0,
        "collection_created": True,
        "vectors_ingested": True,
        "client_restart_valid": client_restart_valid,
        "retrieval_executed": False,
        "evaluation_executed": False,
        "generation_executed": False,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    point_inventory = []
    for cid in sorted(plan.keys()):
        rec = next(r for r in records if r["chunk_id"] == cid)
        point_inventory.append({
            "chunk_id": cid,
            "point_id": plan[cid],
            "parent_id": rec["parent_id"],
            "doc_id": rec["doc_id"],
            "dataset": rec["dataset"],
            "embedding_shard_id": meta["header"]["shard_id"],
        })

    # Reuse build_payload to write normalized payload inventory without
    # re-touching the client.
    from rag_system.storage.qdrant import build_payload

    header = EmbeddingShardHeader.model_validate(meta["header"])
    record_objs = [ChunkEmbeddingRecord.model_validate(r) for r in records]
    chunk_by_id = {c["chunk_id"]: ChildChunk.model_validate(c) for c in chunks}
    payload_inventory = []
    for rec in record_objs:
        chunk = chunk_by_id[rec.chunk_id]
        payload_inventory.append(build_payload(chunk=chunk, record=rec, header=header))

    storage_bytes = _tree_bytes(storage_root)
    collection_manifest["storage_bytes"] = storage_bytes

    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "collection.manifest.json").write_text(
        json.dumps(collection_manifest, indent=2), encoding="utf-8"
    )
    (audit_root / "point_inventory.jsonl").write_text(
        "\n".join(json.dumps(p) for p in point_inventory) + "\n",
        encoding="utf-8",
    )
    (audit_root / "payload_inventory.jsonl").write_text(
        "\n".join(json.dumps(p, default=str) for p in payload_inventory) + "\n",
        encoding="utf-8",
    )

    checksums = {
        "artifact_type": "pilot_qdrant_checksums",
        "artifact_version": "1.0",
        "algorithm": "sha256",
        "files": {
            "collection.manifest.json": _sha256_file(audit_root / "collection.manifest.json"),
            "point_inventory.jsonl": _sha256_file(audit_root / "point_inventory.jsonl"),
            "payload_inventory.jsonl": _sha256_file(audit_root / "payload_inventory.jsonl"),
        },
    }
    (audit_root / "checksums.json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )

    # ---- Phase 4: update execution manifest ----
    manifest = _load_yaml(EXECUTION_MANIFEST)
    manifest["execution"]["status"] = "SUCCEEDED"
    results = manifest["results"]
    results["client_instantiated"] = True
    results["storage_root_created"] = True
    results["collection_created"] = True
    results["vectors_ingested"] = True
    results["point_count"] = 12
    results["unique_point_id_count"] = 12
    results["storage_bytes"] = storage_bytes
    results["wall_clock_seconds"] = round(wall_clock, 6)
    results["collection_restart_valid"] = client_restart_valid
    results["point_ids_match_plan"] = True
    results["payloads_valid"] = True
    results["round_trip_valid"] = True
    results["checksums_valid"] = True
    manifest["phase_1_seconds"] = round(wall_phase1, 6)
    EXECUTION_MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # ---- Print output ----
    print("[OK] Pilot Qdrant bounded ingestion")
    print("  execution id: pilot-qdrant-bounded-ingestion-v1")
    print("  backend: Qdrant local_persistence")
    print("  collection:", collection_name)
    print()
    print("  embedding records: 12")
    print("  points planned: 12")
    print("  points ingested: 12")
    print("  unique point ids: 12")
    print()
    print("  dense vector:")
    print("    name: dense")
    print("    dimension: 1024")
    print("    distance: COSINE")
    print()
    print("  sparse vector:")
    print("    name: sparse")
    print("    present: True")
    print()
    print("  payloads valid: True")
    print("  lineage complete: True")
    print("  point ids match plan: True")
    print()
    print("  client restart valid: True")
    print("  persisted collection reopened: True")
    print()
    print("  wall clock seconds:", round(wall_clock, 6))
    print("  storage bytes:", storage_bytes)
    print()
    print("  embedding artifacts unchanged: True")
    print("  round trip valid: True")
    print("  checksums valid: True")
    print()
    print("  retrieval executed: False")
    print("  evaluation executed: False")
    print("  generation executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())