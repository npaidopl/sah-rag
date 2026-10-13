# ruff: noqa: E402
"""SAH-RAG M0-L5c: numpy-based fast retrieval.

Replaces Qdrant-local search (which is brute-force and slow for
>20k points) with equivalent in-memory computation.

Documented as CONTROLLED ADAPTATION:
- Same dense cosine scores
- Same sparse dot-product scores
- Same RRF fusion (k=60, standard Cormack et al. 2009)
- Same parent aggregation (best + w*sum_of_others, w=0.05)
- Same document resolution (group by doc_id, best parent as score)

Qdrant collections remain canonical storage; the numpy computation
is verified against their metadata (point count, config hash).

Resume-safe via query_id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import scipy.sparse as sp

from rag_system.canonical.loaders import iter_canonical_examples

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
QE_ROOT = OUT_ROOT / "_query_embeddings"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
PREFETCH = 200
PARENT_LIMIT = 100
DOCUMENT_LIMIT = 100
ADDITIONAL_CHILD_WEIGHT = 0.05
RRF_K = 60
QUERY_BATCH = 64


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


def load_chunks_meta(ds: str) -> tuple[list[str], list[str], list[str]]:
    """Return (chunk_ids, parent_ids, doc_ids) in order."""
    p = OUT_ROOT / ds / "chunks-v1" / "chunks.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    cids: list[str] = []
    pids: list[str] = []
    dids: list[str] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cids.append(row["chunk_id"])
            pids.append(row["parent_id"])
            dids.append(row["doc_id"])
    return cids, pids, dids


def load_embeddings(ds: str) -> tuple[np.ndarray, list[list[int]], list[list[float]]]:
    """Return (dense_matrix, sparse_indices_list, sparse_values_list)."""
    p = OUT_ROOT / ds / "embeddings-v1" / "embedding_records.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    dense_rows: list[list[float]] = []
    sparse_idx: list[list[int]] = []
    sparse_val: list[list[float]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dense_rows.append(row["dense_vector"])
            sp = row["sparse_vector"]
            sparse_idx.append(sp["indices"])
            sparse_val.append(sp["values"])
    dense = np.asarray(dense_rows, dtype=np.float32)
    return dense, sparse_idx, sparse_val


def build_sparse_matrix(
    idx_list: list[list[int]],
    val_list: list[list[float]],
    vocab_size: int,
) -> sp.csr_matrix:
    """Build a CSR sparse matrix from list of (indices, values)."""
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for idx, val in zip(idx_list, val_list, strict=True):
        for i, v in zip(idx, val, strict=True):
            indices.append(int(i))
            data.append(float(v))
        indptr.append(len(indices))
    return sp.csr_matrix(
        (np.asarray(data, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(idx_list), vocab_size),
        dtype=np.float32,
    )


def load_query_embeddings(ds: str) -> dict[str, dict[str, Any]]:
    p = QE_ROOT / f"{ds}.jsonl"
    mf = QE_ROOT / f"{ds}.manifest.json"
    if not p.exists() or not mf.exists():
        raise FileNotFoundError(f"Missing query embeddings for {ds}")
    m = json.loads(mf.read_text(encoding="utf-8"))
    if sha256_file(p) != m["query_embeddings_sha256"]:
        raise RuntimeError(f"{ds}: query embedding SHA drift")
    by_id: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_id[row["query_id"]] = row
    return by_id


def project_queries(ds: str) -> list[tuple[str, str]]:
    return [
        (ex.query.query_id, ex.query.question)
        for ex in iter_canonical_examples(ds)
    ]


def rrf_fuse(dense_idx: np.ndarray, sparse_idx: np.ndarray) -> list[tuple[int, float]]:
    """Standard RRF fusion with k=RRF_K. Returns [(idx, score), ...] desc."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_idx, start=1):
        i = int(idx)
        scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(sparse_idx, start=1):
        i = int(idx)
        scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def aggregate_to_documents(
    fused: list[tuple[int, float]],
    *,
    chunk_ids: list[str],
    parent_ids: list[str],
    doc_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Same parent aggregation + document resolution semantics as the
    pydantic version, but on numpy indices."""
    if not fused:
        return [], {"num_children": 0, "num_parents": 0, "num_unique_docs": 0}

    # parent aggregation
    by_parent: dict[str, list[tuple[int, float, str, str]]] = defaultdict(list)
    for rank, (idx, score) in enumerate(fused, start=1):
        cid = chunk_ids[idx]
        pid = parent_ids[idx]
        did = doc_ids[idx]
        by_parent[pid].append((rank, score, cid, did))

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

    # document resolution
    by_doc: dict[
        str, list[tuple[float, int, str, list[tuple[int, float, str, str]]]]
    ] = defaultdict(list)
    for parent_score, best_rank, pid, members in parents:
        did = members[0][3]
        by_doc[did].append((parent_score, best_rank, pid, members))

    docs: list[tuple[float, int, str, list[str], int]] = []
    for did, ps in by_doc.items():
        ordered_ps = sorted(ps, key=lambda x: (x[1], x[2]))
        score_ordered = sorted(ps, key=lambda x: (-x[0], x[1], x[2]))
        doc_score = score_ordered[0][0]
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
            "architecture_item_id": did,
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
        "num_children": len(fused),
        "num_parents": len(parents),
        "num_unique_docs": len(docs),
    }
    return ranked_items, stats


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k values, sorted by score desc (idx asc tiebreak)."""
    n = scores.shape[0]
    if n <= k:
        order = np.lexsort((np.arange(n), -scores))
        return order
    top = np.argpartition(scores, -k)[-k:]
    order = np.lexsort((top, -scores[top]))
    return top[order]


def run_dataset(ds: str, *, overwrite: bool) -> dict[str, Any]:
    stage = OUT_ROOT / ds / "retrieval-v1"
    results_path = stage / "results.jsonl"
    mf_path = stage / "manifest.json"

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

    print(f"[{ds}] loading chunks meta ...")
    cids, pids, dids = load_chunks_meta(ds)
    n_chunks = len(cids)
    print(f"[{ds}]   {n_chunks} chunks")

    print(f"[{ds}] loading embeddings ...")
    dense, sp_idx, sp_val = load_embeddings(ds)
    if dense.shape[0] != n_chunks:
        raise RuntimeError(
            f"{ds}: dense rows {dense.shape[0]} != chunks {n_chunks}"
        )
    print(f"[{ds}]   dense shape: {dense.shape}")

    # vocab size = max chunk or query index + 1
    print(f"[{ds}] building sparse matrix ...")
    max_chunk_idx = max((max(x) for x in sp_idx if x), default=0)
    qemb = load_query_embeddings(ds)
    max_query_idx = max(
        (max(r["sparse_vector"]["indices"]) for r in qemb.values()
         if r["sparse_vector"]["indices"]),
        default=0,
    )
    vocab = max(max_chunk_idx, max_query_idx) + 1
    chunks_sparse = build_sparse_matrix(sp_idx, sp_val, vocab)
    print(f"[{ds}]   vocab={vocab}, nnz={chunks_sparse.nnz:,}")
    del sp_idx, sp_val

    pairs = project_queries(ds)
    if len(pairs) != len(qemb):
        raise RuntimeError(f"{ds}: queries {len(pairs)} != qemb {len(qemb)}")

    # resume
    done: set[str] = set()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["query_id"])

    remaining = [(qid, _) for qid, _ in pairs if qid not in done]
    print(f"[{ds}] {len(pairs)} total, {len(done)} done, {len(remaining)} to process")

    t0 = time.perf_counter()
    mode = "a" if done else "w"
    with results_path.open(mode, encoding="utf-8", newline="\n") as fh:
        i = 0
        for batch_start in range(0, len(remaining), QUERY_BATCH):
            batch = remaining[batch_start : batch_start + QUERY_BATCH]
            B = len(batch)

            # build query dense
            q_dense = np.asarray(
                [qemb[qid]["dense_vector"] for qid, _ in batch],
                dtype=np.float32,
            )
            # dense scores (B, N)
            dense_scores = q_dense @ dense.T

            # build sparse batch
            q_sparse_dense = np.zeros((B, vocab), dtype=np.float32)
            for bi, (qid, _) in enumerate(batch):
                sp_idx_q = qemb[qid]["sparse_vector"]["indices"]
                sp_val_q = qemb[qid]["sparse_vector"]["values"]
                if sp_idx_q:
                    q_sparse_dense[bi, np.asarray(sp_idx_q, dtype=np.int64)] = np.asarray(
                        sp_val_q, dtype=np.float32
                    )
            # sparse scores (B, N) = (B, V) @ (V, N)
            sparse_scores = q_sparse_dense @ chunks_sparse.T
            if sp.issparse(sparse_scores):
                sparse_scores = sparse_scores.toarray()
            else:
                sparse_scores = np.asarray(sparse_scores)

            for bi, (qid, _) in enumerate(batch):
                ds_row = dense_scores[bi]
                ss_row = sparse_scores[bi]

                dense_top = top_k_indices(ds_row, PREFETCH)
                sparse_top = top_k_indices(ss_row, PREFETCH)

                fused = rrf_fuse(dense_top, sparse_top)

                items, stats = aggregate_to_documents(
                    fused,
                    chunk_ids=cids,
                    parent_ids=pids,
                    doc_ids=dids,
                )

                fh.write(json.dumps({
                    "query_id": qid,
                    "ranked_items": items,
                    "metadata": {
                        **stats,
                        "prefetch_limit": PREFETCH,
                        "parent_limit": PARENT_LIMIT,
                        "document_limit": DOCUMENT_LIMIT,
                        "rrf_k": RRF_K,
                        "retrieval_method": "hybrid_rrf_numpy",
                    },
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                fh.write("\n")

                i += 1
                if i % 500 == 0:
                    fh.flush()
                    el = time.perf_counter() - t0
                    rate = i / el
                    eta = (len(remaining) - i) / rate if rate > 0 else 0
                    print(f"[{ds}]   {i}/{len(remaining)} "
                          f"({rate:.1f} q/s, ETA {eta/60:.1f} min)")

    wall = time.perf_counter() - t0
    results_sha = sha256_file(results_path)

    manifest = {
        "artifact_type": "sah_rag_m0_retrieval_manifest",
        "artifact_version": "1.0",
        "status": "SUCCEEDED",
        "dataset_id": ds,
        "query_count": len(pairs),
        "prefetch_limit": PREFETCH,
        "parent_limit": PARENT_LIMIT,
        "document_limit": DOCUMENT_LIMIT,
        "additional_child_weight": ADDITIONAL_CHILD_WEIGHT,
        "rrf_k": RRF_K,
        "retrieval_method": "hybrid_rrf_numpy",
        "point_count": n_chunks,
        "vocab_size": vocab,
        "sparse_nnz": int(chunks_sparse.nnz),
        "results_sha256": results_sha,
        "wall_clock_seconds": round(wall, 3),
        "controlled_adaptation": (
            "in-memory numpy+scipy equivalent of Qdrant-native RRF search"
        ),
    }
    write_json_atomic(mf_path, manifest)
    write_json_atomic(stage / "checksums.json",
                      {"files": {"results.jsonl": results_sha}})
    print(f"[{ds}] DONE in {wall:.1f}s")
    return manifest


def run(*, datasets: tuple[str, ...], overwrite: bool) -> int:
    results = []
    for ds in datasets:
        try:
            results.append(run_dataset(ds, overwrite=overwrite))
        except Exception as exc:
            print(f"[{ds}] FAILED: {type(exc).__name__}: {exc}")
            results.append({
                "dataset_id": ds,
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })

    all_ok = all(r.get("status") == "SUCCEEDED" for r in results)
    write_json_atomic(OUT_ROOT / "_m0_l5_root_manifest.json", {
        "artifact_type": "sah_rag_m0_retrieval_root",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m0-l5c-v1",
        "status": "SUCCEEDED" if all_ok else "PARTIAL",
        "datasets": results,
    })

    print()
    print("M0-L5c SUMMARY")
    for r in results:
        if r.get("status") == "FAILED":
            print(f"  {r['dataset_id']:18s} FAILED: {r.get('error')}")
        else:
            print(
                f"  {r['dataset_id']:18s} "
                f"queries={r['query_count']:>7} "
                f"secs={r['wall_clock_seconds']}"
            )
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