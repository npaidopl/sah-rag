# ruff: noqa: E402
"""LongRAG L4 - Qdrant storage build and reopen verification.

For each dataset in the reduced scope (clerk, cuad, greek_bar_bench,
lara):
  1. Load embeddings-v1/embedding_records.jsonl
  2. Verify count matches embeddings-v1/manifest.json
  3. Create Qdrant collection longrag-<ds>-v1 in local mode
  4. Ingest all points
  5. Close client
  6. Reopen client as a fresh instance
  7. Verify point count matches
  8. Write storage manifest + checksums

Local Qdrant instance lives at outputs/external_longrag/_qdrant/ and
holds all four collections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qdrant_client import QdrantClient

from rag_system.baselines.longrag_embeddings import LongUnitEmbeddingRecord
from rag_system.baselines.longrag_storage import (
    create_longrag_collection,
    ingest_longrag_records,
)

OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
QDRANT_PATH = OUTPUT_ROOT / "_qdrant"

EXPECTED_DATASETS: tuple[str, ...] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

COLLECTION_PREFIX = "longrag"
DENSE_DIMENSION = 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(data))


def load_records(
    dataset_id: str,
) -> tuple[list[LongUnitEmbeddingRecord], dict[str, Any]]:
    stage = OUTPUT_ROOT / dataset_id / "embeddings-v1"
    manifest_path = stage / "manifest.json"
    records_path = stage / "embedding_records.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"Missing records: {records_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = int(manifest["record_count"])

    records: list[LongUnitEmbeddingRecord] = []
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(
                    LongUnitEmbeddingRecord.model_validate_json(line)
                )

    if len(records) != expected_count:
        raise RuntimeError(
            f"{dataset_id}: record count {len(records)} != expected "
            f"{expected_count}"
        )

    actual_sha = sha256_file(records_path)
    if actual_sha != manifest["embedding_records_sha256"]:
        raise RuntimeError(
            f"{dataset_id}: embedding_records.jsonl SHA-256 drift "
            f"({actual_sha} != {manifest['embedding_records_sha256']})"
        )

    return records, manifest


def collection_name_for(dataset_id: str) -> str:
    return f"{COLLECTION_PREFIX}-{dataset_id}-v1"


def ingest_dataset(
    client: QdrantClient,
    dataset_id: str,
    *,
    recreate: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    records, manifest = load_records(dataset_id)
    col = collection_name_for(dataset_id)

    print(f"[{dataset_id}] loaded {len(records)} records from embeddings-v1")

    if recreate and client.collection_exists(col):
        print(f"[{dataset_id}] recreating collection {col}")
        client.delete_collection(col)

    created = create_longrag_collection(
        client=client,
        collection_name=col,
        dense_dimension=DENSE_DIMENSION,
    )
    print(f"[{dataset_id}] collection '{col}' created={created}")

    if created:
        total = ingest_longrag_records(
            client=client,
            collection_name=col,
            records=records,
        )
        print(f"[{dataset_id}] ingested {total} points")
    else:
        existing = client.count(col).count
        if existing == len(records):
            print(
                f"[{dataset_id}] collection exists with {existing} "
                f"points; skipping ingest (idempotent)"
            )
            total = existing
        else:
            print(
                f"[{dataset_id}] collection exists with {existing} "
                f"points; re-ingesting to reach {len(records)}"
            )
            total = ingest_longrag_records(
                client=client,
                collection_name=col,
                records=records,
            )

    count = client.count(col).count
    if count != len(records):
        raise RuntimeError(
            f"{dataset_id}: post-ingest count {count} != {len(records)}"
        )

    wall = time.perf_counter() - t0
    return {
        "dataset_id": dataset_id,
        "collection_name": col,
        "expected_count": len(records),
        "ingested_count": total,
        "verified_count": count,
        "embedding_records_sha256": manifest["embedding_records_sha256"],
        "wall_clock_seconds": round(wall, 3),
    }


def run(*, datasets: tuple[str, ...], recreate: bool) -> int:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[L4] Qdrant local path: {QDRANT_PATH}")

    results: list[dict[str, Any]] = []

    print("\n[L4] PHASE 1: create collections and ingest")
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        for ds in datasets:
            try:
                results.append(
                    ingest_dataset(client, ds, recreate=recreate)
                )
            except Exception as exc:
                print(f"[{ds}] FAILED: {exc}")
                results.append(
                    {
                        "dataset_id": ds,
                        "status": "FAILED",
                        "error": str(exc),
                    }
                )
    finally:
        client.close()

    print("\n[L4] PHASE 2: reopen and verify")
    client2 = QdrantClient(path=str(QDRANT_PATH))
    try:
        for r in results:
            if r.get("status") == "FAILED":
                continue
            col = r["collection_name"]
            count = client2.count(col).count
            r["reopened_count"] = count
            r["reopen_verified"] = (count == r["expected_count"])
            print(
                f"[{r['dataset_id']}] reopened count={count} "
                f"verified={r['reopen_verified']}"
            )
    finally:
        client2.close()

    all_ok = all(
        r.get("reopen_verified", False) for r in results
    )

    manifest = {
        "artifact_type": "longrag_full_track_qdrant_storage_manifest",
        "artifact_version": "1.0",
        "execution_id": "longrag-l4-storage-v1",
        "status": "SUCCEEDED" if all_ok else "PARTIAL",
        "qdrant_path": str(QDRANT_PATH),
        "datasets": results,
        "next_gate": "longrag_full_track_retrieval",
    }
    write_json_atomic(QDRANT_PATH / "storage_manifest.json", manifest)

    print()
    print("L4 SUMMARY")
    for r in results:
        if r.get("status") == "FAILED":
            print(f"  {r['dataset_id']:18s} FAILED: {r.get('error')}")
        else:
            print(
                f"  {r['dataset_id']:18s} "
                f"points={r['verified_count']:>6} "
                f"reopen={r['reopen_verified']}"
            )
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(EXPECTED_DATASETS),
        choices=list(EXPECTED_DATASETS),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate existing collections.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(datasets=tuple(args.datasets), recreate=args.recreate)


if __name__ == "__main__":
    raise SystemExit(main())