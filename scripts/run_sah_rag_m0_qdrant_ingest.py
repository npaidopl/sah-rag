# ruff: noqa: E402
"""SAH-RAG M0-L4 - Qdrant hybrid storage build + reopen verification.

For each dataset in the 4-dataset scope:
  1. Load embeddings-v1/embedding_records.jsonl
  2. Load chunks-v1/chunks.jsonl
  3. Create hybrid Qdrant collection (dense 1024 COSINE + sparse IP)
  4. Ingest all points via sah_rag.storage.qdrant.ingest_embedding_records
  5. Close client, reopen, verify point count

Local Qdrant instance at outputs/external_sah_rag/_qdrant/.
Collection naming: myrag-<dataset>-v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qdrant_client import QdrantClient

from sah_rag.artifacts.models import ChildChunk
from sah_rag.embedding.models import ChunkEmbeddingRecord
from sah_rag.storage.qdrant import (
    create_hybrid_collection,
    ingest_embedding_records,
)

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
QDRANT_PATH = OUT_ROOT / "_qdrant"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
DENSE_DIMENSION = 1024


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json_atomic(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def collection_name_for(ds: str) -> str:
    return f"myrag-{ds}-v1"


def load_records(ds: str) -> tuple[list[ChunkEmbeddingRecord], dict[str, Any]]:
    stage = OUT_ROOT / ds / "embeddings-v1"
    recs_path = stage / "embedding_records.jsonl"
    mf_path = stage / "manifest.json"
    if not recs_path.exists() or not mf_path.exists():
        raise FileNotFoundError(f"Missing embeddings for {ds}")

    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    expected = int(mf["record_count"])

    records: list[ChunkEmbeddingRecord] = []
    with recs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            raw.pop("metadata", None)
            records.append(ChunkEmbeddingRecord.model_validate(raw))

    if len(records) != expected:
        raise RuntimeError(
            f"{ds}: record count {len(records)} != manifest {expected}"
        )

    actual_sha = sha256_file(recs_path)
    if actual_sha != mf["embedding_records_sha256"]:
        raise RuntimeError(f"{ds}: embedding_records SHA drift")

    return records, mf


def load_chunks(ds: str) -> tuple[list[ChildChunk], dict[str, Any]]:
    stage = OUT_ROOT / ds / "chunks-v1"
    chunks_path = stage / "chunks.jsonl"
    mf_path = stage / "manifest.json"
    if not chunks_path.exists() or not mf_path.exists():
        raise FileNotFoundError(f"Missing chunks for {ds}")

    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    expected = int(mf["chunk_count"])

    chunks: list[ChildChunk] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(ChildChunk.model_validate_json(line))

    if len(chunks) != expected:
        raise RuntimeError(
            f"{ds}: chunk count {len(chunks)} != manifest {expected}"
        )
    return chunks, mf


def ingest_dataset(
    client: QdrantClient, ds: str, *, recreate: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    records, emb_mf = load_records(ds)
    chunks, chunks_mf = load_chunks(ds)
    print(f"[{ds}] loaded {len(records)} records, {len(chunks)} chunks")

    if len(records) != len(chunks):
        raise RuntimeError(
            f"{ds}: records {len(records)} != chunks {len(chunks)}"
        )

    col = collection_name_for(ds)

    if recreate and client.collection_exists(col):
        print(f"[{ds}] recreating {col}")
        client.delete_collection(col)

    if not client.collection_exists(col):
        create_hybrid_collection(
            client=client,
            collection_name=col,
            dense_dimension=DENSE_DIMENSION,
        )
        print(f"[{ds}] collection '{col}' created")

    # Build a minimal EmbeddingShardHeader for the ingestion helper
    from sah_rag.embedding.models import (
        EmbeddingKind,
        EmbeddingShardHeader,
        VectorPrecision,
    )
    header = EmbeddingShardHeader(
        shard_id=f"myrag-{ds}-shard-0",
        shard_index=0,
        dataset=ds,
        model_id=emb_mf["model_id"],
        model_revision=emb_mf["model_revision"],
        tokenizer_id=emb_mf["model_id"],
        embedding_config_hash=emb_mf["embedding_config_hash"],
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=DENSE_DIMENSION,
        dense_precision=VectorPrecision.FLOAT16,
        dense_normalized=True,
        maximum_records=max(len(records), 1),
    )

    existing = client.count(col, exact=True).count
    if existing == len(records):
        print(f"[{ds}] already has {existing} points; skipping ingest")
        total = existing
    else:
        print(f"[{ds}] ingesting {len(records)} points ...")
        summary = ingest_embedding_records(
            client=client,
            collection_name=col,
            header=header,
            records=records,
            chunks=chunks,
            batch_size=128,
        )
        total = summary.point_count
        print(f"[{ds}] ingested {total} points")

    count = client.count(col, exact=True).count
    if count != len(records):
        raise RuntimeError(
            f"{ds}: post-ingest count {count} != {len(records)}"
        )

    wall = time.perf_counter() - t0
    return {
        "dataset_id": ds,
        "collection_name": col,
        "expected_count": len(records),
        "ingested_count": total,
        "verified_count": count,
        "embedding_records_sha256": emb_mf["embedding_records_sha256"],
        "chunks_jsonl_sha256": chunks_mf["chunks_jsonl_sha256"],
        "chunker_config_hash": chunks_mf["chunker_config_hash"],
        "embedding_config_hash": emb_mf["embedding_config_hash"],
        "wall_clock_seconds": round(wall, 3),
    }


def run(*, datasets: tuple[str, ...], recreate: bool) -> int:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[M0-L4] Qdrant local path: {QDRANT_PATH}")

    results: list[dict[str, Any]] = []

    print("\n[M0-L4] PHASE 1: create collections and ingest")
    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        for ds in datasets:
            try:
                results.append(ingest_dataset(client, ds, recreate=recreate))
            except Exception as exc:
                print(f"[{ds}] FAILED: {type(exc).__name__}: {exc}")
                results.append({
                    "dataset_id": ds,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        client.close()

    print("\n[M0-L4] PHASE 2: reopen and verify")
    client2 = QdrantClient(path=str(QDRANT_PATH))
    try:
        for r in results:
            if r.get("status") == "FAILED":
                continue
            col = r["collection_name"]
            count = client2.count(col, exact=True).count
            r["reopened_count"] = count
            r["reopen_verified"] = (count == r["expected_count"])
            print(
                f"[{r['dataset_id']}] reopened count={count} "
                f"verified={r['reopen_verified']}"
            )
    finally:
        client2.close()

    all_ok = all(r.get("reopen_verified", False) for r in results)

    manifest = {
        "artifact_type": "sah_rag_m0_qdrant_storage_manifest",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m0-l4-v1",
        "status": "SUCCEEDED" if all_ok else "PARTIAL",
        "qdrant_path": str(QDRANT_PATH),
        "datasets": results,
        "next_gate": "sah_rag_m0_query_embeddings",
    }
    write_json_atomic(QDRANT_PATH / "storage_manifest.json", manifest)

    print()
    print("M0-L4 SUMMARY")
    print("=" * 78)
    total_points = 0
    for r in results:
        if r.get("status") == "FAILED":
            print(f"  {r['dataset_id']:18s} FAILED: {r.get('error')}")
        else:
            print(
                f"  {r['dataset_id']:18s} "
                f"points={r['verified_count']:>7} "
                f"reopen={r['reopen_verified']} "
                f"secs={r['wall_clock_seconds']}"
            )
            total_points += r["verified_count"]
    print()
    print(f"TOTAL POINTS: {total_points}")
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--recreate", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    return run(datasets=tuple(args.datasets), recreate=args.recreate)


if __name__ == "__main__":
    raise SystemExit(main())