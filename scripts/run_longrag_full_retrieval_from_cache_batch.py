# ruff: noqa: E402
"""LongRAG L5b - Full retrieval using pre-computed query embeddings.

Reads query embeddings from outputs/external_longrag/_query_embeddings/
(produced on Colab GPU) instead of encoding on CPU.

Same output schema as L5 (results.jsonl) so that L6 evaluation is
unchanged. Ground truth never loaded here.
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
from rag_system.baselines.longrag_retrieval import (
    retrieve_long_units,
    retrieved_to_ranked_long_units,
)
from rag_system.canonical.loaders import iter_canonical_examples

OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
QDRANT_PATH = OUTPUT_ROOT / "_qdrant"
CACHE_DIR = OUTPUT_ROOT / "_query_embeddings"

EXPECTED_DATASETS: tuple[str, ...] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

TOP_K_UNITS = 100
DOCUMENT_LIMIT = 100
BGE_M3_DENSE_DIMENSION = 1024


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


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canonical_json_bytes(data))
    tmp.replace(path)


def project_queries(dataset_id: str) -> list[tuple[str, str]]:
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
    return pairs


def load_cached_query_embeddings(
    dataset_id: str,
    pairs: list[tuple[str, str]],
) -> tuple[dict[str, tuple[float, ...]], dict[str, Any]]:
    cache_path = CACHE_DIR / f"{dataset_id}.jsonl"
    manifest_path = CACHE_DIR / f"{dataset_id}.manifest.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache: {cache_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_id"] != dataset_id:
        raise RuntimeError(
            f"{dataset_id}: manifest dataset_id mismatch"
        )

    actual_sha = sha256_file(cache_path)
    if actual_sha != manifest["sha256"]:
        raise RuntimeError(
            f"{dataset_id}: cache SHA-256 drift "
            f"({actual_sha} != {manifest['sha256']})"
        )

    qvec_by_id: dict[str, tuple[float, ...]] = {}
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            vec = tuple(float(x) for x in row["dense_vector"])
            if len(vec) != BGE_M3_DENSE_DIMENSION:
                raise RuntimeError(
                    f"{dataset_id}: vector dim {len(vec)} for {qid}"
                )
            qvec_by_id[qid] = vec

    expected_ids = {p[0] for p in pairs}
    cached_ids = set(qvec_by_id.keys())
    if cached_ids != expected_ids:
        missing = expected_ids - cached_ids
        extra = cached_ids - expected_ids
        raise RuntimeError(
            f"{dataset_id}: query_id set mismatch. "
            f"missing={len(missing)}, extra={len(extra)}. "
            f"sample missing={list(missing)[:3]}, "
            f"sample extra={list(extra)[:3]}"
        )

    if manifest["record_count"] != len(pairs):
        raise RuntimeError(
            f"{dataset_id}: manifest count {manifest['record_count']} "
            f"!= pairs {len(pairs)}"
        )

    return qvec_by_id, manifest


def retrieve_all(
    dataset_id: str,
    pairs: list[tuple[str, str]],
    qvec_by_id: dict[str, tuple[float, ...]],
    cache_manifest: dict[str, Any],
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

    stage.mkdir(parents=True, exist_ok=True)
    results_path = stage / "results.jsonl"

    print(f"[{dataset_id}] retrieving for {len(pairs)} queries ...")
    t0 = time.perf_counter()

    with results_path.open("w", encoding="utf-8", newline="\n") as fh:
        for i, (qid, _qtext) in enumerate(pairs):
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
                ranked_items.append({
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
                })

            fh.write(json.dumps({
                "query_id": qid,
                "ranked_items": ranked_items,
                "metadata": {
                    "num_units_retrieved": len(retrieved),
                    "num_unique_docs": len(expanded),
                    "top_k_units": TOP_K_UNITS,
                    "document_limit": DOCUMENT_LIMIT,
                },
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            fh.write("\n")

            if (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[{dataset_id}]   {i+1}/{len(pairs)} "
                    f"({(i+1)/max(elapsed,1e-9):.1f} q/s)"
                )

    wall = time.perf_counter() - t0
    results_sha = sha256_file(results_path)

    manifest: dict[str, Any] = {
        "artifact_type": "longrag_full_track_retrieval_manifest",
        "artifact_version": "1.0",
        "execution_id": f"longrag-l5b-retrieval-{dataset_id}-v1",
        "status": "SUCCEEDED",
        "dataset_id": dataset_id,
        "collection_name": col,
        "query_count": len(pairs),
        "top_k_units": TOP_K_UNITS,
        "document_limit": DOCUMENT_LIMIT,
        "query_embedding_source": "precomputed_cache",
        "query_embedding_config_hash": cache_manifest["embedding_config_hash"],
        "query_precision": cache_manifest["precision"],
        "query_device": cache_manifest["device"],
        "query_max_length": cache_manifest["max_length"],
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


def run(*, datasets: tuple[str, ...]) -> int:
    if not QDRANT_PATH.exists():
        print(f"MISSING qdrant: {QDRANT_PATH}")
        return 2
    if not CACHE_DIR.exists():
        print(f"MISSING cache dir: {CACHE_DIR}")
        return 2

    client = QdrantClient(path=str(QDRANT_PATH))
    results: list[dict[str, Any]] = []
    try:
        for ds in datasets:
            try:
                pairs = project_queries(ds)
                print(f"[{ds}] {len(pairs)} queries loaded")
                qvec_by_id, cache_manifest = load_cached_query_embeddings(
                    ds, pairs,
                )
                print(f"[{ds}] cache hit: {len(qvec_by_id)} vectors")
                manifest = retrieve_all(
                    ds, pairs, qvec_by_id, cache_manifest, client,
                )
                results.append(manifest)
            except Exception as exc:
                print(f"[{ds}] FAILED: {exc}")
                results.append({
                    "dataset_id": ds, "status": "FAILED", "error": str(exc),
                })
    finally:
        client.close()

    all_ok = all(r.get("status") == "SUCCEEDED" for r in results)
    root = {
        "artifact_type": "longrag_full_track_retrieval_manifest",
        "artifact_version": "1.0",
        "execution_id": "longrag-l5b-all-v1",
        "status": "SUCCEEDED" if all_ok else "PARTIAL",
        "datasets": results,
    }
    write_json_atomic(OUTPUT_ROOT / "_l5_root_manifest.json", root)

    print()
    print("L5b SUMMARY")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(datasets=tuple(args.datasets))


if __name__ == "__main__":
    raise SystemExit(main())