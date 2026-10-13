# ruff: noqa: E402
"""SAH-RAG M0-L5b β€” Fast retrieval (raw dicts, resume-safe).

Replaces pydantic-heavy path with raw dict manipulation. Uses
payload field selection to avoid fetching text_normalized. Writes
append-only results.jsonl so a restart resumes from the last
completed query.

Retrieval logic (dense + sparse + RRF + parent aggregation +
document resolution) is functionally identical to the original
implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qdrant_client import QdrantClient, models as qm

from rag_system.canonical.loaders import iter_canonical_examples

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
QDRANT_PATH = OUT_ROOT / "_qdrant"
QUERY_EMB_ROOT = OUT_ROOT / "_query_embeddings"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
CANDIDATE_LIMIT = 200
PARENT_LIMIT = 100
DOCUMENT_LIMIT = 100
ADDITIONAL_CHILD_WEIGHT = 0.05
PAYLOAD_FIELDS = ["chunk_id", "parent_id", "doc_id"]


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


def project_queries(ds: str) -> list[tuple[str, str]]:
    return [
        (ex.query.query_id, ex.query.question)
        for ex in iter_canonical_examples(ds)
    ]


def load_query_embeddings(ds: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    recs = QUERY_EMB_ROOT / f"{ds}.jsonl"
    mf = QUERY_EMB_ROOT / f"{ds}.manifest.json"
    if not recs.exists() or not mf.exists():
        raise FileNotFoundError(f"Missing query embeddings for {ds}")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    if sha256_file(recs) != manifest["query_embeddings_sha256"]:
        raise RuntimeError(f"{ds}: query embeddings SHA drift")
    by_id: dict[str, dict[str, Any]] = {}
    with recs.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r["query_id"]] = r
    return by_id, manifest


def retrieve_one(
    client: QdrantClient,
    col: str,
    dense: list[float],
    sidx: list[int],
    sval: list[float],
) -> list[tuple[str, str, float]]:
    """Return list of (chunk_id, parent_id, doc_id, score) after RRF."""
    response = client.query_points(
        collection_name=col,
        prefetch=[
            qm.Prefetch(
                query=dense, using="dense", limit=CANDIDATE_LIMIT,
            ),
            qm.Prefetch(
                query=qm.SparseVector(indices=sidx, values=sval),
                using="sparse", limit=CANDIDATE_LIMIT,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=CANDIDATE_LIMIT,
        with_payload=PAYLOAD_FIELDS,
    )

    rows: list[tuple[str, str, str, float]] = []
    for p in response.points:
        pl = p.payload or {}
        cid = pl.get("chunk_id")
        pid = pl.get("parent_id")
        did = pl.get("doc_id")
        if not (isinstance(cid, str) and isinstance(pid, str) and isinstance(did, str)):
            raise RuntimeError(f"Missing payload fields in point {p.id}")
        rows.append((cid, pid, did, float(p.score)))
    return rows


def aggregate_to_documents(
    rows: list[tuple[str, str, str, float]],
    *,
    query_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Aggregate children -> parents -> documents. Raw dicts."""
    # Rows are already ranked by RRF (index = rank-1)
    # Parent aggregation: rank within parent list = position in rows.
    by_parent: dict[str, list[tuple[int, float, str, str]]] = defaultdict(list)
    for idx, (cid, pid, did, score) in enumerate(rows):
        by_parent[pid].append((idx + 1, score, cid, did))

    parents: list[tuple[float, int, str, list[tuple[int, float, str, str]]]] = []
    for pid, members in by_parent.items():
        ordered = sorted(members, key=lambda m: (m[0], -m[1], m[2]))
        score_ordered = sorted(members, key=lambda m: (-m[1], m[0], m[2]))
        best = score_ordered[0]
        additional = sum(m[1] for m in score_ordered[1:])
        parent_score = best[1] + ADDITIONAL_CHILD_WEIGHT * additional
        parents.append((parent_score, ordered[0][0], pid, ordered))

    parents.sort(key=lambda p: (-p[0], p[1], p[2]))
    parents = parents[:PARENT_LIMIT]

    by_doc: dict[str, list[tuple[float, int, str, list[tuple[int, float, str, str]]]]] = defaultdict(list)
    for parent_score, best_rank, pid, members in parents:
        did = members[0][3]
        by_doc[did].append((parent_score, best_rank, pid, members))

    docs: list[tuple[float, int, str, list[str], int]] = []
    for did, ps in by_doc.items():
        ordered_ps = sorted(ps, key=lambda x: (x[1], x[2]))
        score_ordered_ps = sorted(ps, key=lambda x: (-x[0], x[1], x[2]))
        best_parent = score_ordered_ps[0]
        doc_score = best_parent[0]
        best_rank = ordered_ps[0][1]

        seen: set[str] = set()
        all_chunks: list[str] = []
        for _, _, _, members in ordered_ps:
            for _, _, cid, _ in members:
                if cid not in seen:
                    seen.add(cid)
                    all_chunks.append(cid)

        docs.append((doc_score, best_rank, did, all_chunks, len(ps)))

    docs.sort(key=lambda d: (-d[0], d[1], d[2]))
    docs = docs[:DOCUMENT_LIMIT]

    ranked_items = []
    for rank, (score, best_rank, did, chunks, n_parents) in enumerate(docs, start=1):
        ranked_items.append({
            "architecture_item_id": did,  # doc as the item for SAH-RAG
            "canonical_doc_id": did,
            "canonical_chunk_or_evidence_ids": chunks,
            "rank": rank,
            "score": float(score),
            "score_semantics": "HIGHER_IS_BETTER",
            "metadata": {
                "best_parent_rank": best_rank,
                "num_contributing_parents": n_parents,
                "num_contributing_chunks": len(chunks),
            },
        })

    stats = {
        "num_children": len(rows),
        "num_parents": len(by_parent),
        "num_unique_docs": len(docs),
    }
    return ranked_items, stats


def run_dataset(
    ds: str,
    client: QdrantClient,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    stage = OUT_ROOT / ds / "retrieval-v1"
    results_path = stage / "results.jsonl"
    mf_path = stage / "manifest.json"
    progress_path = stage / "progress.json"

    if mf_path.exists() and not overwrite:
        try:
            m = json.loads(mf_path.read_text(encoding="utf-8"))
            if m.get("status") == "SUCCEEDED":
                print(f"[{ds}] already SUCCEEDED; skip")
                return m
        except json.JSONDecodeError:
            pass

    stage.mkdir(parents=True, exist_ok=True)
    if overwrite and results_path.exists():
        results_path.unlink()
    if overwrite and progress_path.exists():
        progress_path.unlink()

    # Resume: how many queries already done?
    done_ids: set[str] = set()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["query_id"])

    pairs = project_queries(ds)
    qemb, _ = load_query_embeddings(ds)
    col = f"myrag-{ds}-v1"
    if not client.collection_exists(col):
        raise RuntimeError(f"Missing collection {col}")

    remaining = [(qid, _) for qid, _ in pairs if qid not in done_ids]
    missing_emb = [qid for qid, _ in remaining if qid not in qemb]
    if missing_emb:
        raise RuntimeError(f"{ds}: {len(missing_emb)} queries lack embeddings")

    print(f"[{ds}] {len(pairs)} total, {len(done_ids)} already done, "
          f"{len(remaining)} to process")

    t0 = time.perf_counter()
    append_mode = "a" if done_ids else "w"

    with results_path.open(append_mode, encoding="utf-8", newline="\n") as fh:
        for i, (qid, _) in enumerate(remaining, start=1):
            row = qemb[qid]
            dense = row["dense_vector"]
            sp = row["sparse_vector"]
            sidx = sp["indices"]
            sval = sp["values"]

            rows = retrieve_one(client, col, dense, sidx, sval)
            items, stats = aggregate_to_documents(rows, query_id=qid)

            fh.write(json.dumps({
                "query_id": qid,
                "ranked_items": items,
                "metadata": {
                    **stats,
                    "candidate_limit": CANDIDATE_LIMIT,
                    "parent_limit": PARENT_LIMIT,
                    "document_limit": DOCUMENT_LIMIT,
                    "retrieval_method": "hybrid_rrf",
                },
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            fh.write("\n")

            if i % 500 == 0:
                fh.flush()
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                eta = (len(remaining) - i) / rate if rate > 0 else 0
                print(f"[{ds}]   {i}/{len(remaining)} ({rate:.1f} q/s, ETA {eta/60:.1f} min)")

    wall = time.perf_counter() - t0
    results_sha = sha256_file(results_path)

    manifest = {
        "artifact_type": "sah_rag_m0_retrieval_manifest",
        "artifact_version": "1.0",
        "status": "SUCCEEDED",
        "dataset_id": ds,
        "collection_name": col,
        "query_count": len(pairs),
        "candidate_limit": CANDIDATE_LIMIT,
        "parent_limit": PARENT_LIMIT,
        "document_limit": DOCUMENT_LIMIT,
        "additional_child_weight": ADDITIONAL_CHILD_WEIGHT,
        "retrieval_method": "hybrid_rrf",
        "results_sha256": results_sha,
        "wall_clock_seconds": round(wall, 3),
    }
    write_json_atomic(mf_path, manifest)
    write_json_atomic(stage / "checksums.json",
                      {"files": {"results.jsonl": results_sha}})
    print(f"[{ds}] done in {wall:.1f}s")
    return manifest


def run(*, datasets: tuple[str, ...], overwrite: bool) -> int:
    if not QDRANT_PATH.exists():
        print("MISSING qdrant")
        return 2
    if not QUERY_EMB_ROOT.exists():
        print("MISSING query embeddings")
        return 2

    client = QdrantClient(path=str(QDRANT_PATH))
    results = []
    try:
        for ds in datasets:
            try:
                results.append(run_dataset(ds, client, overwrite=overwrite))
            except Exception as exc:
                print(f"[{ds}] FAILED: {type(exc).__name__}: {exc}")
                results.append({"dataset_id": ds, "status": "FAILED",
                                "error": str(exc)})
    finally:
        client.close()

    all_ok = all(r.get("status") == "SUCCEEDED" for r in results)
    write_json_atomic(OUT_ROOT / "_m0_l5_root_manifest.json", {
        "artifact_type": "sah_rag_m0_retrieval_root",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m0-l5b-v1",
        "status": "SUCCEEDED" if all_ok else "PARTIAL",
        "datasets": results,
    })

    print()
    print("M0-L5b SUMMARY")
    for r in results:
        if r.get("status") == "FAILED":
            print(f"  {r['dataset_id']:18s} FAILED: {r.get('error')}")
        else:
            print(f"  {r['dataset_id']:18s} queries={r['query_count']:>7} secs={r['wall_clock_seconds']}")
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    return run(datasets=tuple(args.datasets), overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())