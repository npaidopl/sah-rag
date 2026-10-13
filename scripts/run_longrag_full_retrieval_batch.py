# ruff: noqa: E402
"""LongRAG L5 - Full retrieval across the 4-dataset scope.

For each dataset:
  1. Load examples.jsonl
  2. Project queries to (query_id, query_text) only
  3. Encode queries locally with pinned BGE-M3 (FP32, CPU)
  4. Retrieve top-100 long units per query from Qdrant
  5. Expand ranked long units -> ranked canonical documents (dedup)
  6. Write retrieval-v1/results.jsonl per query
  7. Write manifest + checksums

Ground truth is NEVER loaded in this script. Query text only.
Resume-safe: if retrieval-v1/manifest.json is SUCCEEDED, skip.
"""
from __future__ import annotations
# --- SAH-RAG path resolution (sanitized from local absolute paths) ---
import os as _os
from pathlib import Path as _Path
_SAH_RAG_REPO_ROOT = _Path(__file__).resolve().parents[1]
_SAH_RAG_PROJECT_ROOT = _os.environ.get(
    "SAH_RAG_PROJECT_ROOT",
    str(_SAH_RAG_REPO_ROOT.parent / "Project 1"),
)
_SAH_RAG_CANONICAL_ROOT = _os.environ.get(
    "SAH_RAG_CANONICAL_ROOT",
    str(_Path(_SAH_RAG_PROJECT_ROOT) / "processed" / "canonical"),
)
# ---------------------------------------------------------------------

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qdrant_client import QdrantClient

from rag_system.baselines.longrag import (
    expand_ranked_long_units_to_documents,
)
from rag_system.baselines.longrag_bge_m3 import (
    BGE_M3_DENSE_DIMENSION,
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BgeM3DenseEncoder,
)
from rag_system.baselines.longrag_retrieval import (
    retrieve_long_units,
    retrieved_to_ranked_long_units,
)
from rag_system.canonical.loaders import iter_canonical_examples

OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
QDRANT_PATH = OUTPUT_ROOT / "_qdrant"
QUERY_EMB_ROOT = OUTPUT_ROOT / "_query_embeddings"

SNAPSHOT_PATH = Path(
    "D:/rag_work/hf_cache/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

EXPECTED_DATASETS: tuple[str, ...] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

TOP_K_UNITS = 100
DOCUMENT_LIMIT = 100
QUERY_BATCH_SIZE = 8
QUERY_MAX_LENGTH = 512

EMBEDDING_CONFIG_HASH_QUERY = hashlib.sha256(
    f"{BGE_M3_MODEL_ID}:{BGE_M3_MODEL_REVISION}:dense:fp32:cpu:query".encode()
).hexdigest()


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


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                )
            )
            fh.write("\n")
    tmp.replace(path)


def project_queries(dataset_id: str) -> tuple[list[tuple[str, str]], str]:
    """Return [(query_id, query_text)] and the examples file SHA-256."""
    examples_path = (
        Path(_SAH_RAG_CANONICAL_ROOT)
        / dataset_id
        / "examples.jsonl"
    )
    if not examples_path.exists():
        raise FileNotFoundError(f"Missing examples: {examples_path}")

    pairs: list[tuple[str, str]] = []
    for ex in iter_canonical_examples(dataset_id):
        qid = ex.query.query_id
        qtext = ex.query.question
        if not qid.strip() or not qtext.strip():
            raise RuntimeError(
                f"{dataset_id}: empty query_id or question in examples.jsonl"
            )
        pairs.append((qid, qtext))

    if not pairs:
        raise RuntimeError(f"{dataset_id}: no queries found")

    return pairs, sha256_file(examples_path)


def encode_queries(
    dataset_id: str,
    pairs: list[tuple[str, str]],
    encoder: BgeM3DenseEncoder,
) -> Path:
    """Encode all queries and cache to JSONL. Returns cache path."""
    QUERY_EMB_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = QUERY_EMB_ROOT / f"{dataset_id}.jsonl"
    cache_meta = QUERY_EMB_ROOT / f"{dataset_id}.manifest.json"

    if cache_path.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta.get("record_count") == len(pairs):
            if meta.get("sha256") == sha256_file(cache_path):
                print(f"[{dataset_id}] query embedding cache hit")
                return cache_path

    print(f"[{dataset_id}] encoding {len(pairs)} queries (FP32 CPU) ...")
    t0 = time.perf_counter()

    query_texts = [p[1] for p in pairs]
    rows: list[dict[str, Any]] = []

    for start in range(0, len(query_texts), QUERY_BATCH_SIZE):
        batch = query_texts[start : start + QUERY_BATCH_SIZE]
        vecs = encoder.encode_dense(
            batch,
            batch_size=QUERY_BATCH_SIZE,
            max_length=QUERY_MAX_LENGTH,
        )
        for j, vec in enumerate(vecs):
            idx = start + j
            qid = pairs[idx][0]
            if len(vec) != BGE_M3_DENSE_DIMENSION:
                raise RuntimeError(
                    f"Query vector dim {len(vec)} != {BGE_M3_DENSE_DIMENSION}"
                )
            rows.append(
                {
                    "query_id": qid,
                    "dense_vector": list(vec),
                    "dense_dimension": BGE_M3_DENSE_DIMENSION,
                    "dense_normalized": True,
                    "model_id": BGE_M3_MODEL_ID,
                    "model_revision": BGE_M3_MODEL_REVISION,
                    "embedding_config_hash": EMBEDDING_CONFIG_HASH_QUERY,
                    "precision": "float32",
                    "device": "cpu",
                }
            )
        if (start // QUERY_BATCH_SIZE) % 20 == 0:
            elapsed = time.perf_counter() - t0
            done = start + len(batch)
            rate = done / max(elapsed, 1e-9)
            print(
                f"[{dataset_id}]   {done}/{len(query_texts)} "
                f"({rate:.1f} q/s)"
            )

    write_jsonl_atomic(cache_path, rows)

    meta = {
        "dataset_id": dataset_id,
        "record_count": len(rows),
        "sha256": sha256_file(cache_path),
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "embedding_config_hash": EMBEDDING_CONFIG_HASH_QUERY,
        "wall_clock_seconds": round(time.perf_counter() - t0, 3),
    }
    write_json_atomic(cache_meta, meta)
    print(
        f"[{dataset_id}] query encoding done in "
        f"{meta['wall_clock_seconds']}s"
    )
    return cache_path


def retrieve_all(
    dataset_id: str,
    pairs: list[tuple[str, str]],
    query_emb_path: Path,
    client: QdrantClient,
) -> dict[str, Any]:
    stage = OUTPUT_ROOT / dataset_id / "retrieval-v1"
    manifest_path = stage / "manifest.json"

    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            if m.get("status") == "SUCCEEDED":
                print(f"[{dataset_id}] retrieval already SUCCEEDED; skip")
                return m
        except json.JSONDecodeError:
            pass

    col = f"longrag-{dataset_id}-v1"
    if not client.collection_exists(col):
        raise RuntimeError(f"Collection {col} does not exist")

    print(f"[{dataset_id}] loading query embeddings from cache")
    qvec_by_id: dict[str, tuple[float, ...]] = {}
    with query_emb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qvec_by_id[row["query_id"]] = tuple(row["dense_vector"])

    if len(qvec_by_id) != len(pairs):
        raise RuntimeError(
            f"{dataset_id}: query vector count {len(qvec_by_id)} != "
            f"pairs {len(pairs)}"
        )

    stage.mkdir(parents=True, exist_ok=True)
    results_path = stage / "results.jsonl"

    print(f"[{dataset_id}] retrieving for {len(pairs)} queries ...")
    t0 = time.perf_counter()

    with results_path.open("w", encoding="utf-8", newline="\n") as fh:
        for i, (qid, qtext) in enumerate(pairs):
            qvec = qvec_by_id[qid]
            retrieved = retrieve_long_units(
                client=client,
                collection_name=col,
                query_vector=qvec,
                limit=TOP_K_UNITS,
            )
            ranked_units = retrieved_to_ranked_long_units(
                query_id=qid, retrieved=retrieved,
            )
            expanded = expand_ranked_long_units_to_documents(
                query_id=qid,
                ranked_long_units=ranked_units,
                document_limit=DOCUMENT_LIMIT,
            )

            ranked_items: list[dict[str, Any]] = []
            for doc in expanded:
                ranked_items.append(
                    {
                        "architecture_item_id": doc.source_long_unit_id,
                        "canonical_doc_id": doc.doc_id,
                        "canonical_chunk_or_evidence_ids": [],
                        "rank": doc.document_rank,
                        "score": float(doc.source_long_unit_score),
                        "score_semantics": "HIGHER_IS_BETTER",
                        "metadata": {
                            "source_long_unit_rank": doc.source_long_unit_rank,
                            "membership_position": doc.membership_position,
                        },
                    }
                )

            fh.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "ranked_items": ranked_items,
                        "metadata": {
                            "num_units_retrieved": len(retrieved),
                            "num_unique_docs": len(expanded),
                            "top_k_units": TOP_K_UNITS,
                            "document_limit": DOCUMENT_LIMIT,
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            fh.write("\n")

            if (i + 1) % 200 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                print(
                    f"[{dataset_id}]   {i+1}/{len(pairs)} "
                    f"({rate:.1f} q/s)"
                )

    wall = time.perf_counter() - t0
    results_sha = sha256_file(results_path)

    manifest: dict[str, Any] = {
        "artifact_type": "longrag_full_track_retrieval_manifest",
        "artifact_version": "1.0",
        "execution_id": f"longrag-l5-retrieval-{dataset_id}-v1",
        "status": "SUCCEEDED",
        "dataset_id": dataset_id,
        "collection_name": col,
        "query_count": len(pairs),
        "top_k_units": TOP_K_UNITS,
        "document_limit": DOCUMENT_LIMIT,
        "query_embedding_config_hash": EMBEDDING_CONFIG_HASH_QUERY,
        "query_precision": "float32",
        "query_device": "cpu",
        "results_sha256": results_sha,
        "wall_clock_seconds": round(wall, 3),
        "next_gate": "longrag_full_track_evaluation",
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        stage / "checksums.json",
        {"files": {"results.jsonl": results_sha}},
    )
    print(f"[{dataset_id}] retrieval done in {wall:.1f}s")
    return manifest


def run(*, datasets: tuple[str, ...], force: bool) -> int:
    if not SNAPSHOT_PATH.exists():
        print(f"MISSING snapshot: {SNAPSHOT_PATH}")
        return 2
    if not QDRANT_PATH.exists():
        print(f"MISSING qdrant: {QDRANT_PATH}")
        return 2

    print(f"[L5] loading BGE-M3 (CPU, FP32) ...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    encoder = BgeM3DenseEncoder(
        snapshot_path=SNAPSHOT_PATH, device="cpu", use_fp16=False,
    )
    print("[L5] encoder ready")

    client = QdrantClient(path=str(QDRANT_PATH))
    results: list[dict[str, Any]] = []
    try:
        for ds in datasets:
            try:
                pairs, ex_sha = project_queries(ds)
                print(f"[{ds}] {len(pairs)} queries loaded")
                emb_path = encode_queries(ds, pairs, encoder)
                manifest = retrieve_all(ds, pairs, emb_path, client)
                manifest["examples_file_sha256"] = ex_sha
                results.append(manifest)
            except Exception as exc:
                print(f"[{ds}] FAILED: {exc}")
                results.append(
                    {"dataset_id": ds, "status": "FAILED", "error": str(exc)}
                )
    finally:
        client.close()

    all_ok = all(r.get("status") == "SUCCEEDED" for r in results)
    root_manifest = {
        "artifact_type": "longrag_full_track_retrieval_manifest",
        "artifact_version": "1.0",
        "execution_id": "longrag-l5-all-v1",
        "status": "SUCCEEDED" if all_ok else "PARTIAL",
        "datasets": results,
    }
    write_json_atomic(OUTPUT_ROOT / "_l5_root_manifest.json", root_manifest)

    print()
    print("L5 SUMMARY")
    for r in results:
        if r.get("status") == "FAILED":
            print(f"  {r['dataset_id']:18s} FAILED: {r.get('error')}")
        else:
            print(
                f"  {r['dataset_id']:18s} "
                f"queries={r['query_count']:>6} "
                f"secs={r['wall_clock_seconds']}"
            )
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="*",
        default=list(EXPECTED_DATASETS),
        choices=list(EXPECTED_DATASETS),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(datasets=tuple(args.datasets), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())