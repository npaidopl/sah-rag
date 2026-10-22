# ruff: noqa: E402
"""SAH-RAG M5 - Confirmatory evaluation on the full dev split.

Evaluates a frozen list of confirmatory configs on the FULL dev split
(not just the 10% sample used in M4A). Reports macro metrics per
config per dataset, with per-query results cached so we can compute
paired bootstrap uncertainty and Pareto selection in M6.

Configs are selected from the M4B reduction: the four
non-dominated configs from dominance analysis, plus coverage
alternatives (sparse, larger-chunk, default).
"""
from __future__ import annotations

import argparse
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

from sah_rag.canonical.loaders import iter_canonical_examples

OUT_ROOT = REPO / "outputs" / "external_sah_rag"
QE_ROOT = OUT_ROOT / "_query_embeddings"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
CUTOFFS: tuple[int, ...] = (1, 5, 10, 20, 50)
PRECISION_K = 10
NDCG_CUTOFFS: tuple[int, ...] = (10, 50)
RRF_K = 60
ADDITIONAL_CHILD_WEIGHT = 0.05
PARENT_LIMIT = 100
DOCUMENT_LIMIT = 100

# Confirmatory set: 8 configs covering best/diverse.
CONFIRMATORY: tuple[tuple[int, int, str, int], ...] = (
    (256, 32, "hybrid_rrf", 200),   # frozen default
    (256, 128, "hybrid_rrf", 200),  # best primary metric
    (256, 32, "hybrid_rrf", 100),   # best primary with lower k
    (256, 128, "hybrid_rrf", 100),  # best primary with lower k
    (256, 32, "sparse", 200),       # best sparse
    (256, 128, "sparse", 200),      # best sparse alt
    (512, 32, "hybrid_rrf", 200),   # larger chunk alt
    (512, 128, "hybrid_rrf", 200),  # larger chunk alt
)


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


def load_ground_truth(
    ds: str, qids: set[str],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ex in iter_canonical_examples(ds):
        if ex.query.query_id in qids:
            out[ex.query.query_id] = list(ex.ground_truth.relevant_ids)
    return out


def load_chunks_meta(
    ds: str, cs: int, ov: int,
) -> tuple[list[str], list[str], list[str]]:
    p = OUT_ROOT / ds / "grid-chunks-v1" / f"c{cs}_o{ov}" / "chunks.jsonl"
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


def load_dense(ds: str, cs: int, ov: int) -> np.ndarray:
    p = (
        OUT_ROOT / ds / "grid-embeddings-v1" / f"c{cs}_o{ov}"
        / "embedding_records.jsonl"
    )
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
            sp_obj = row["sparse_vector"]
            idx_list.append(sp_obj["indices"])
            val_list.append(sp_obj["values"])
    return idx_list, val_list


def build_csr(
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


def load_query_emb(
    ds: str, qids: list[str],
) -> tuple[np.ndarray, list[list[int]], list[list[float]]]:
    p = QE_ROOT / f"{ds}.jsonl"
    by_id: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_id[row["query_id"]] = row
    dense = np.asarray(
        [by_id[q]["dense_vector"] for q in qids], dtype=np.float32,
    )
    sidx = [by_id[q]["sparse_vector"]["indices"] for q in qids]
    sval = [by_id[q]["sparse_vector"]["values"] for q in qids]
    return dense, sidx, sval


def batch_topk(scores: np.ndarray, k: int) -> np.ndarray:
    Q, N = scores.shape
    k_eff = min(k, N)
    if k_eff == N:
        return np.argsort(-scores, axis=1, kind="stable").astype(np.int32)
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
    parent_ids: list[str],
    doc_ids: list[str],
) -> list[str]:
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


def macro_avg(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def run_config(
    ds: str,
    sample_ids: list[str],
    gt: dict[str, list[str]],
    q_dense: np.ndarray,
    q_sidx: list[list[int]],
    q_sval: list[list[float]],
    cs: int,
    ov: int,
    mode: str,
    K: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cids, pids, dids = load_chunks_meta(ds, cs, ov)
    dense = load_dense(ds, cs, ov)
    sidx_list, sval_list = load_sparse_lists(ds, cs, ov)

    max_cs = 0
    for lst in sidx_list:
        if lst:
            max_cs = max(max_cs, max(lst))
    max_q = 0
    for lst in q_sidx:
        if lst:
            max_q = max(max_q, max(lst))
    vocab = max(max_cs, max_q) + 1

    sparse_csr = build_csr(sidx_list, sval_list, vocab)
    del sidx_list, sval_list
    q_sparse = build_csr(q_sidx, q_sval, vocab)

    dense_scores = q_dense @ dense.T
    sp_scores_raw = q_sparse @ sparse_csr.T
    sparse_scores = (
        sp_scores_raw.toarray()
        if sp.issparse(sp_scores_raw)
        else np.asarray(sp_scores_raw)
    )

    dense_topk = batch_topk(dense_scores, K)
    sparse_topk = batch_topk(sparse_scores, K)

    per_query: list[dict[str, Any]] = []
    for qi, qid in enumerate(sample_ids):
        if mode == "dense":
            idx_arr = dense_topk[qi]
            fused = [
                (int(idx_arr[i]),
                 float(dense_scores[qi, idx_arr[i]]))
                for i in range(len(idx_arr))
            ]
        elif mode == "sparse":
            idx_arr = sparse_topk[qi]
            fused = [
                (int(idx_arr[i]),
                 float(sparse_scores[qi, idx_arr[i]]))
                for i in range(len(idx_arr))
            ]
        else:
            fused = rrf_fuse(
                dense_topk[qi].tolist(),
                sparse_topk[qi].tolist(),
                k=RRF_K,
            )[:K]

        ranked = aggregate_to_documents(
            fused, parent_ids=pids, doc_ids=dids,
        )
        metrics = evaluate_query(ranked, gt[qid])
        per_query.append({"query_id": qid, **metrics})

    macro = macro_avg(per_query)
    return macro, per_query


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    t_start = time.perf_counter()
    dev = load_dev_split()

    results: list[dict[str, Any]] = []
    per_query_all: list[dict[str, Any]] = []

    for ds in args.datasets:
        sample_ids = dev[ds]
        print(f"\n[{ds}] full dev = {len(sample_ids)} queries")
        gt = load_ground_truth(ds, set(sample_ids))
        q_dense, q_sidx, q_sval = load_query_emb(ds, sample_ids)

        # Cache dense/sparse chunk matrices per (cs, ov) — reuse
        # across different (mode, K) for the same variant.
        variants = sorted({(cs, ov) for cs, ov, _, _ in CONFIRMATORY})
        caches: dict[tuple[int, int], dict[str, Any]] = {}

        for cs, ov in variants:
            print(f"[{ds}]   building caches for c{cs}_o{ov} ...")
            cids, pids, dids = load_chunks_meta(ds, cs, ov)
            dense = load_dense(ds, cs, ov)
            sidx_list, sval_list = load_sparse_lists(ds, cs, ov)
            max_cs = 0
            for lst in sidx_list:
                if lst:
                    max_cs = max(max_cs, max(lst))
            max_q = 0
            for lst in q_sidx:
                if lst:
                    max_q = max(max_q, max(lst))
            vocab = max(max_cs, max_q) + 1
            sparse_csr = build_csr(sidx_list, sval_list, vocab)
            del sidx_list, sval_list
            q_sparse = build_csr(q_sidx, q_sval, vocab)
            dense_scores = q_dense @ dense.T
            sp_raw = q_sparse @ sparse_csr.T
            sparse_scores = (
                sp_raw.toarray() if sp.issparse(sp_raw)
                else np.asarray(sp_raw)
            )
            caches[(cs, ov)] = {
                "cids": cids, "pids": pids, "dids": dids,
                "dense_scores": dense_scores,
                "sparse_scores": sparse_scores,
            }
            del dense, sparse_csr, q_sparse, sp_raw

        for cs, ov, mode, K in CONFIRMATORY:
            cache = caches[(cs, ov)]
            dense_scores = cache["dense_scores"]
            sparse_scores = cache["sparse_scores"]
            pids = cache["pids"]
            dids = cache["dids"]
            dense_topk = batch_topk(dense_scores, K)
            sparse_topk = batch_topk(sparse_scores, K)

            t0 = time.perf_counter()
            per_query: list[dict[str, float]] = []
            for qi, qid in enumerate(sample_ids):
                if mode == "dense":
                    idx_arr = dense_topk[qi]
                    fused = [
                        (int(idx_arr[i]),
                         float(dense_scores[qi, idx_arr[i]]))
                        for i in range(len(idx_arr))
                    ]
                elif mode == "sparse":
                    idx_arr = sparse_topk[qi]
                    fused = [
                        (int(idx_arr[i]),
                         float(sparse_scores[qi, idx_arr[i]]))
                        for i in range(len(idx_arr))
                    ]
                else:
                    fused = rrf_fuse(
                        dense_topk[qi].tolist(),
                        sparse_topk[qi].tolist(),
                        k=RRF_K,
                    )[:K]
                ranked = aggregate_to_documents(
                    fused, parent_ids=pids, doc_ids=dids,
                )
                m = evaluate_query(ranked, gt[qid])
                per_query.append(m)
                per_query_all.append({
                    "dataset": ds,
                    "config_id": config_id(cs, ov, mode, K),
                    "query_id": qid,
                    **m,
                })

            macro = macro_avg(per_query)
            elapsed = time.perf_counter() - t0
            cid = config_id(cs, ov, mode, K)
            print(
                f"[{ds}]   {cid:35s} R@10={macro['recall@10']:.4f} "
                f"({elapsed:.1f}s)"
            )

            results.append({
                "dataset": ds,
                "config_id": cid,
                "chunk_size": cs,
                "chunk_overlap": ov,
                "retrieval_mode": mode,
                "candidate_limit": K,
                "query_count": len(sample_ids),
                "macro": macro,
                "elapsed_seconds": round(elapsed, 3),
            })

    # write results
    res_path = OUT_ROOT / "_m5_confirmatory_results.jsonl"
    with res_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")

    pq_path = OUT_ROOT / "_m5_per_query_metrics.jsonl"
    with pq_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in per_query_all:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")

    # macro across datasets per config
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_cfg[r["config_id"]].append(r)
    macro_rows: list[dict[str, Any]] = []
    for cid, rows in by_cfg.items():
        if len(rows) != len(args.datasets):
            continue
        primary = float(np.mean([r["macro"]["recall@10"] for r in rows]))
        h10 = float(np.mean([r["macro"]["hit@10"] for r in rows]))
        nd = float(np.mean([r["macro"]["ndcg@10"] for r in rows]))
        macro_rows.append({
            "config_id": cid,
            "macro_recall@10": primary,
            "macro_hit@10": h10,
            "macro_ndcg@10": nd,
        })
    macro_rows.sort(key=lambda r: (-r["macro_recall@10"], r["config_id"]))

    manifest = {
        "artifact_type": "sah_rag_m5_confirmatory_manifest",
        "artifact_version": "1.0",
        "execution_id": "sah-rag-m5-v1",
        "status": "SUCCEEDED",
        "row_count": len(results),
        "per_query_row_count": len(per_query_all),
        "configs": [c for c in by_cfg],
        "config_count": len(by_cfg),
        "datasets": list(args.datasets),
        "wall_clock_seconds": round(time.perf_counter() - t_start, 3),
        "macro_table": macro_rows,
    }
    write_json_atomic(OUT_ROOT / "_m5_confirmatory_manifest.json", manifest)

    print()
    print("=" * 90)
    print("M5 CONFIRMATORY — MACRO ACROSS DATASETS (full dev split)")
    print("=" * 90)
    print(f"{'config_id':40s} {'R@10':>8s} {'H@10':>8s} {'nDCG@10':>8s}")
    for r in macro_rows:
        print(
            f"{r['config_id']:40s} "
            f"{r['macro_recall@10']:>8.4f} "
            f"{r['macro_hit@10']:>8.4f} "
            f"{r['macro_ndcg@10']:>8.4f}"
        )

    print()
    print(
        f"M5 COMPLETE in {manifest['wall_clock_seconds']:.1f}s "
        f"({len(results)} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())