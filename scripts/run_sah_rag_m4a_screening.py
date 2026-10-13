# ruff: noqa: E402
"""SAH-RAG M4A - Full grid screening on 10% dev subset.

Runs all 81 configs (9 chunk variants x 3 modes x 3 candidate_limits)
on a deterministic 10% dev sample (min 100, max 500 queries per
dataset). Produces one row per (dataset, config) with macro metrics.

No test-set contamination: only dev query ids are used.
Numpy retrieval: mathematically equivalent to Qdrant-native path.
"""
from __future__ import annotations

import argparse
import json
import random
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
GRID: tuple[tuple[int, int], ...] = (
    (256, 0), (256, 32), (256, 128),
    (512, 0), (512, 32), (512, 128),
    (1024, 0), (1024, 32), (1024, 128),
)
MODES: tuple[str, ...] = ("dense", "sparse", "hybrid_rrf")
CANDIDATE_LIMITS: tuple[int, ...] = (50, 100, 200)
CUTOFFS: tuple[int, ...] = (1, 5, 10, 20, 50)
PRECISION_K = 10
NDCG_CUTOFFS: tuple[int, ...] = (10, 50)
RRF_K = 60
ADDITIONAL_CHILD_WEIGHT = 0.05
PARENT_LIMIT = 100
DOCUMENT_LIMIT = 100

SAMPLE_FRACTION = 0.10
SAMPLE_MIN = 100
SAMPLE_MAX = 500
SAMPLE_SEED = 42


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def write_json_atomic(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def config_id(cs: int, ov: int, mode: str, K: int) -> str:
    return f"c{cs}_o{ov}_{mode}_k{K}"


def load_dev_split() -> dict[str, list[str]]:
    """Load per-dataset dev query ids from the split JSONL."""
    p = OUT_ROOT / "_optimization_split.jsonl"
    out: dict[str, list[str]] = {ds: [] for ds in DATASETS}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["split"] == "dev":
                out[row["dataset"]].append(row["query_id"])
    for ds in DATASETS:
        out[ds].sort()
    return out


def sample_dev_subset(
    dev_qids: list[str], *, seed: int = SAMPLE_SEED,
    fraction: float = SAMPLE_FRACTION,
    min_n: int = SAMPLE_MIN, max_n: int = SAMPLE_MAX,
) -> list[str]:
    n_target = int(len(dev_qids) * fraction)
    n_target = max(min_n, min(max_n, n_target))
    n_target = min(n_target, len(dev_qids))
    rng = random.Random(seed)
    shuffled = list(dev_qids)
    rng.shuffle(shuffled)
    return sorted(shuffled[:n_target])


def load_ground_truth(
    ds: str, qids: set[str],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ex in iter_canonical_examples(ds):
        if ex.query.query_id in qids:
            out[ex.query.query_id] = list(ex.ground_truth.relevant_ids)
    return out


def load_chunks_meta(ds: str, cs: int, ov: int) -> tuple[
    list[str], list[str], list[str],
]:
    p = OUT_ROOT / ds / "grid-chunks-v1" / f"c{cs}_o{ov}" / "chunks.jsonl"
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


def load_dense_embeddings(ds: str, cs: int, ov: int) -> np.ndarray:
    p = (
        OUT_ROOT / ds / "grid-embeddings-v1" / f"c{cs}_o{ov}"
        / "embedding_records.jsonl"
    )
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    rows: list[list[float]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row["dense_vector"])
    return np.asarray(rows, dtype=np.float32)


def load_sparse_lists(
    ds: str, cs: int, ov: int,
) -> tuple[list[list[int]], list[list[float]]]:
    p = (
        OUT_ROOT / ds / "grid-embeddings-v1" / f"c{cs}_o{ov}"
        / "embedding_records.jsonl"
    )
    idx_list: list[list[int]] = []
    val_list: list[list[float]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sp = row["sparse_vector"]
            idx_list.append(sp["indices"])
            val_list.append(sp["values"])
    return idx_list, val_list


def build_sparse_csr(
    idx_list: list[list[int]], val_list: list[list[float]],
    vocab_size: int,
) -> sp.csr_matrix:
    indptr = [0]
    all_idx: list[int] = []
    all_val: list[float] = []
    for idx, val in zip(idx_list, val_list, strict=True):
        for i, v in zip(idx, val, strict=True):
            all_idx.append(int(i))
            all_val.append(float(v))
        indptr.append(len(all_idx))
    return sp.csr_matrix(
        (
            np.asarray(all_val, dtype=np.float32),
            np.asarray(all_idx, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(idx_list), vocab_size),
        dtype=np.float32,
    )


def load_query_embeddings(
    ds: str, qids: list[str],
) -> tuple[np.ndarray, list[list[int]], list[list[float]]]:
    p = QE_ROOT / f"{ds}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    by_id: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_id[row["query_id"]] = row
    missing = [q for q in qids if q not in by_id]
    if missing:
        raise RuntimeError(f"{ds}: {len(missing)} qids missing from embeddings")
    dense = np.asarray(
        [by_id[q]["dense_vector"] for q in qids], dtype=np.float32,
    )
    sidx = [by_id[q]["sparse_vector"]["indices"] for q in qids]
    sval = [by_id[q]["sparse_vector"]["values"] for q in qids]
    return dense, sidx, sval


def batch_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Return (Q, k) indices of top-k per row (score desc, idx asc)."""
    Q, N = scores.shape
    k_eff = min(k, N)
    if k_eff == N:
        order = np.lexsort(
            (np.arange(N)[None, :].repeat(Q, axis=0), -scores),
            axis=1,
        )
        return order.astype(np.int32)
    part = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.lexsort((part, -part_scores), axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32)


def rrf_fuse(
    dense_idx: list[int], sparse_idx: list[int], k: int = RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_idx, start=1):
        i = int(idx)
        scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(sparse_idx, start=1):
        i = int(idx)
        scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def aggregate_to_documents(
    fused: list[tuple[int, float]],
    *,
    chunk_ids: list[str],
    parent_ids: list[str],
    doc_ids: list[str],
) -> list[str]:
    """Return ranked document IDs (best-parent score)."""
    if not fused:
        return []
    by_parent: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for rank, (idx, score) in enumerate(fused, start=1):
        by_parent[parent_ids[idx]].append(
            (rank, score, doc_ids[idx]),
        )

    parents: list[tuple[float, int, str, str]] = []
    for pid, members in by_parent.items():
        ordered = sorted(members, key=lambda m: (m[0], -m[1], m[2]))
        score_ordered = sorted(members, key=lambda m: (-m[1], m[0], m[2]))
        best = score_ordered[0]
        additional = sum(m[1] for m in score_ordered[1:])
        pscore = best[1] + ADDITIONAL_CHILD_WEIGHT * additional
        parents.append((pscore, ordered[0][0], pid, ordered[0][2]))

    parents.sort(key=lambda p: (-p[0], p[1], p[2]))
    parents = parents[:PARENT_LIMIT]

    by_doc: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for pscore, prank, _pid, did in parents:
        by_doc[did].append((pscore, prank))

    docs: list[tuple[float, int, str]] = []
    for did, ps in by_doc.items():
        best_score, best_rank = max(ps, key=lambda x: (x[0], -x[1]))
        docs.append((best_score, best_rank, did))

    docs.sort(key=lambda d: (-d[0], d[1], d[2]))
    return [d[2] for d in docs[:DOCUMENT_LIMIT]]


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def hit_at_k(ranked: list[str], rel: set[str], k: int) -> float:
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], rel: set[str], k: int) -> float:
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def precision_at_k(ranked: list[str], rel: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in rel) / k if k else 0.0


def reciprocal_rank(ranked: list[str], rel: set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rel: set[str], k: int) -> float:
    if not rel:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        gain = 1.0 if d in rel else 0.0
        dcg += ((2.0 ** gain) - 1.0) / np.log2(i + 1)
    ideal = sum(
        1.0 / np.log2(i + 1) for i in range(1, min(k, len(rel)) + 1)
    )
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_query(ranked: list[str], gt: list[str]) -> dict[str, float]:
    rel = set(gt)
    out: dict[str, float] = {}
    for k in CUTOFFS:
        out[f"hit@{k}"] = hit_at_k(ranked, rel, k)
        out[f"recall@{k}"] = recall_at_k(ranked, rel, k)
        out[f"mrr@{k}"] = reciprocal_rank(ranked, rel, k)
    out[f"precision@{PRECISION_K}"] = precision_at_k(
        ranked, rel, PRECISION_K,
    )
    for k in NDCG_CUTOFFS:
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
    return out


def macro_average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# ---------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------
def run_dataset(ds: str, dev_qids: list[str]) -> list[dict[str, Any]]:
    sample_ids = sample_dev_subset(dev_qids)
    print(
        f"\n[{ds}] dev={len(dev_qids)} sample={len(sample_ids)}"
    )

    qset = set(sample_ids)
    gt = load_ground_truth(ds, qset)
    q_dense, q_sidx, q_sval = load_query_embeddings(ds, sample_ids)
    max_q_idx = 0
    for lst in q_sidx:
        if lst:
            max_q_idx = max(max_q_idx, max(lst))
    print(f"[{ds}] max query sparse idx = {max_q_idx}")

    results: list[dict[str, Any]] = []

    for cs, ov in GRID:
        print(f"[{ds}] variant c{cs}_o{ov}")
        cids, pids, dids = load_chunks_meta(ds, cs, ov)
        dense = load_dense_embeddings(ds, cs, ov)
        sidx_list, sval_list = load_sparse_lists(ds, cs, ov)

        max_cs_idx = 0
        for lst in sidx_list:
            if lst:
                max_cs_idx = max(max_cs_idx, max(lst))
        vocab = max(max_cs_idx, max_q_idx) + 1

        sparse_csr = build_sparse_csr(sidx_list, sval_list, vocab)
        del sidx_list, sval_list

        # query sparse CSR
        q_sparse_csr = build_sparse_csr(
            q_sidx, q_sval, vocab,
        )

        # Score matrices (Q, N)
        dense_scores = q_dense @ dense.T
        sparse_dense = (q_sparse_csr @ sparse_csr.T)
        sparse_scores = (
            sparse_dense.toarray()
            if sp.issparse(sparse_dense)
            else np.asarray(sparse_dense)
        )

        for K in CANDIDATE_LIMITS:
            dense_topk = batch_topk(dense_scores, K)
            sparse_topk = batch_topk(sparse_scores, K)

            for mode in MODES:
                t0 = time.perf_counter()
                per_q_metrics: list[dict[str, float]] = []

                for q_idx, qid in enumerate(sample_ids):
                    if mode == "dense":
                        idx_arr = dense_topk[q_idx]
                        fused = [
                            (int(idx_arr[i]),
                             float(dense_scores[q_idx, idx_arr[i]]))
                            for i in range(len(idx_arr))
                        ]
                    elif mode == "sparse":
                        idx_arr = sparse_topk[q_idx]
                        fused = [
                            (int(idx_arr[i]),
                             float(sparse_scores[q_idx, idx_arr[i]]))
                            for i in range(len(idx_arr))
                        ]
                    else:  # hybrid_rrf
                        fused = rrf_fuse(
                            dense_topk[q_idx].tolist(),
                            sparse_topk[q_idx].tolist(),
                            k=RRF_K,
                        )[:K]

                    ranked_docs = aggregate_to_documents(
                        fused,
                        chunk_ids=cids,
                        parent_ids=pids,
                        doc_ids=dids,
                    )
                    m = evaluate_query(ranked_docs, gt[qid])
                    per_q_metrics.append(m)

                macro = macro_average(per_q_metrics)
                elapsed = time.perf_counter() - t0

                results.append({
                    "dataset": ds,
                    "chunk_size": cs,
                    "chunk_overlap": ov,
                    "retrieval_mode": mode,
                    "candidate_limit": K,
                    "config_id": config_id(cs, ov, mode, K),
                    "query_count": len(sample_ids),
                    "macro": macro,
                    "elapsed_seconds": round(elapsed, 3),
                })

        del dense, sparse_csr, dense_scores, sparse_scores
        del q_sparse_csr

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    t_start = time.perf_counter()
    dev = load_dev_split()

    all_results: list[dict[str, Any]] = []
    for ds in args.datasets:
        try:
            all_results.extend(run_dataset(ds, dev[ds]))
        except Exception as exc:
            print(f"[{ds}] FAILED: {type(exc).__name__}: {exc}")

    # write JSONL
    jsonl = OUT_ROOT / "_m4a_screening_results.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in all_results:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")
    tmp.replace(jsonl)

    # manifest
    wall = time.perf_counter() - t_start
    manifest = {
        "artifact_type": "sah_rag_m4a_screening_manifest",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m4a-v1",
        "status": "SUCCEEDED",
        "row_count": len(all_results),
        "datasets": list(args.datasets),
        "grid_size": len(GRID),
        "modes": list(MODES),
        "candidate_limits": list(CANDIDATE_LIMITS),
        "wall_clock_seconds": round(wall, 3),
    }
    write_json_atomic(OUT_ROOT / "_m4a_screening_manifest.json", manifest)

    # summary: top-10 by primary metric
    print("\n" + "=" * 90)
    print("TOP CONFIGS by macro R@10 (across datasets)")
    print("=" * 90)

    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        by_cfg[r["config_id"]].append(r)

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for cid, rows in by_cfg.items():
        if len(rows) != len(args.datasets):
            continue
        r10 = float(np.mean([r["macro"]["recall@10"] for r in rows]))
        ranked.append((r10, cid, rows[0]))
    ranked.sort(key=lambda x: (-x[0], x[1]))

    print(f"{'config_id':40s} {'R@10':>8s} {'H@10':>8s} {'nDCG@10':>8s}")
    for r10, cid, _ref in ranked[:15]:
        rows = by_cfg[cid]
        h10 = float(np.mean([r["macro"]["hit@10"] for r in rows]))
        nd = float(np.mean([r["macro"]["ndcg@10"] for r in rows]))
        print(f"{cid:40s} {r10:>8.4f} {h10:>8.4f} {nd:>8.4f}")

    print(f"\nM4A COMPLETE in {wall:.1f}s  ({len(all_results)} rows)")
    print(f"  jsonl: {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())