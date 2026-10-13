# ruff: noqa: E402
"""My_RAG M7 - Locked test evaluation on the frozen config.

Frozen config: c256_o32_hybrid_rrf_k200 (from M6).
Protocol: run ONCE on the test split. No parameter adjustment
after reading test metrics. Produces final metrics.

Outputs:
  - _m7_locked_test_results.jsonl (per dataset)
  - _m7_locked_test_manifest.json
  - docs/my_rag_m7_locked_test.md
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
import yaml

from rag_system.canonical.loaders import iter_canonical_examples

OUT_ROOT = REPO / "outputs" / "external_my_rag"
QE_ROOT = OUT_ROOT / "_query_embeddings"
BASELINE = REPO / "baseline_audit"
DOCS = REPO / "docs"

DATASETS = ("clerk", "cuad", "greek_bar_bench", "lara")
CUTOFFS: tuple[int, ...] = (1, 5, 10, 20, 50)
PRECISION_K = 10
NDCG_CUTOFFS: tuple[int, ...] = (10, 50)
RRF_K = 60
ADDITIONAL_CHILD_WEIGHT = 0.05
PARENT_LIMIT = 100
DOCUMENT_LIMIT = 100


def load_frozen_config() -> dict[str, Any]:
    p = BASELINE / "my_rag_m6_frozen_config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def load_test_split() -> dict[str, list[str]]:
    p = OUT_ROOT / "_optimization_split.jsonl"
    out: dict[str, list[str]] = {ds: [] for ds in DATASETS}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["split"] == "test":
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


def load_sparse(
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
            obj = row["sparse_vector"]
            idx_list.append(obj["indices"])
            val_list.append(obj["values"])
    return idx_list, val_list


def build_csr(
    idx_list: list[list[int]], val_list: list[list[float]],
    vocab: int,
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
        shape=(len(idx_list), vocab),
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


def write_json_atomic(p: Path, d: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def write_text_atomic(p: Path, t: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(t, encoding="utf-8", newline="\n")
    tmp.replace(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    print("[M7] loading frozen config ...")
    frozen = load_frozen_config()
    sel = frozen["selected_config"]
    cs = sel["chunk_size"]
    ov = sel["chunk_overlap"]
    mode = sel["retrieval_mode"]
    K = sel["candidate_limit"]
    cid = frozen["selected_config_id"]
    print(
        f"[M7] frozen: {cid} "
        f"(cs={cs}, ov={ov}, mode={mode}, K={K})"
    )

    print("[M7] loading test split ...")
    test = load_test_split()

    t_start = time.perf_counter()
    results: list[dict[str, Any]] = []

    for ds in args.datasets:
        test_ids = test[ds]
        print(f"\n[M7] [{ds}] test = {len(test_ids)} queries")
        gt = load_ground_truth(ds, set(test_ids))
        q_dense, q_sidx, q_sval = load_query_emb(ds, test_ids)

        print(f"[M7] [{ds}] loading c{cs}_o{ov} chunks ...")
        _cids, pids, dids = load_chunks_meta(ds, cs, ov)
        dense = load_dense(ds, cs, ov)
        sidx_list, sval_list = load_sparse(ds, cs, ov)

        max_cs_idx = 0
        for lst in sidx_list:
            if lst:
                max_cs_idx = max(max_cs_idx, max(lst))
        max_q_idx = 0
        for lst in q_sidx:
            if lst:
                max_q_idx = max(max_q_idx, max(lst))
        vocab = max(max_cs_idx, max_q_idx) + 1

        sparse_csr = build_csr(sidx_list, sval_list, vocab)
        del sidx_list, sval_list
        q_sparse = build_csr(q_sidx, q_sval, vocab)

        dense_scores = q_dense @ dense.T
        sp_raw = q_sparse @ sparse_csr.T
        sparse_scores = (
            sp_raw.toarray() if sp.issparse(sp_raw)
            else np.asarray(sp_raw)
        )

        dense_topk = batch_topk(dense_scores, K)
        sparse_topk = batch_topk(sparse_scores, K)

        print(f"[M7] [{ds}] evaluating {len(test_ids)} queries ...")
        t0 = time.perf_counter()
        per_query: list[dict[str, float]] = []
        for qi, qid in enumerate(test_ids):
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
        elapsed = time.perf_counter() - t0

        macro = macro_avg(per_query)
        print(
            f"[M7] [{ds}] R@10={macro['recall@10']:.4f} "
            f"({elapsed:.1f}s)"
        )
        results.append({
            "dataset": ds,
            "config_id": cid,
            "query_count": len(test_ids),
            "macro": macro,
            "elapsed_seconds": round(elapsed, 3),
        })

    # macro average
    macro_metrics = macro_avg([r["macro"] for r in results])

    # write per-dataset results
    res_path = OUT_ROOT / "_m7_locked_test_results.jsonl"
    with res_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")

    manifest = {
        "artifact_type": "my_rag_m7_locked_test_manifest",
        "artifact_version": "1.0",
        "execution_id": "my-rag-m7-v1",
        "status": "SUCCEEDED",
        "frozen_config_id": cid,
        "frozen_config": sel,
        "datasets": list(args.datasets),
        "per_dataset_results": results,
        "macro_metrics": macro_metrics,
        "primary_metric": "macro_document_recall@10",
        "primary_metric_value": macro_metrics.get("recall@10"),
        "wall_clock_seconds": round(time.perf_counter() - t_start, 3),
        "protocol_note": (
            "Locked test. Single execution of frozen config on test split. "
            "No parameter adjustment after reading test metrics."
        ),
    }
    write_json_atomic(OUT_ROOT / "_m7_locked_test_manifest.json", manifest)

    # markdown report
    md: list[str] = []
    md.append("# My_RAG M7 — Locked Test Evaluation")
    md.append("")
    md.append(f"**Frozen config**: `{cid}`")
    md.append("")
    md.append(f"- chunk_size = {cs}")
    md.append(f"- chunk_overlap = {ov}")
    md.append(f"- retrieval_mode = {mode}")
    md.append(f"- candidate_limit = {K}")
    md.append("")
    md.append("## Per-dataset results")
    md.append("")
    md.append("| Dataset | Queries | H@10 | R@10 | nDCG@10 | MRR@10 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        m = r["macro"]
        md.append(
            f"| {r['dataset']} | {r['query_count']} | "
            f"{m['hit@10']:.4f} | {m['recall@10']:.4f} | "
            f"{m['ndcg@10']:.4f} | {m['mrr@10']:.4f} |"
        )
    md.append(
        f"| **MACRO** | | {macro_metrics['hit@10']:.4f} | "
        f"{macro_metrics['recall@10']:.4f} | "
        f"{macro_metrics['ndcg@10']:.4f} | "
        f"{macro_metrics['mrr@10']:.4f} |"
    )
    md.append("")
    md.append("## Primary metric")
    md.append("")
    md.append(
        f"`macro_document_recall@10 = {macro_metrics['recall@10']:.6f}`"
    )
    md.append("")
    md.append("## Full metric set (macro)")
    md.append("")
    for k in sorted(macro_metrics.keys()):
        md.append(f"- `{k}` = {macro_metrics[k]:.6f}")
    md.append("")
    md.append("## Protocol")
    md.append("")
    md.append("Locked test: single execution of the frozen config on the "
              "test split. No parameter adjustment was performed after "
              "reading test metrics.")
    write_text_atomic(DOCS / "my_rag_m7_locked_test.md", "\n".join(md))

    print()
    print("=" * 90)
    print("M7 LOCKED TEST — MACRO (frozen config on test split)")
    print("=" * 90)
    print(f"Frozen config: {cid}")
    print()
    print(f"{'dataset':18s} {'queries':>8s} {'H@10':>8s} {'R@10':>8s} "
          f"{'nDCG@10':>8s} {'MRR@10':>8s}")
    for r in results:
        m = r["macro"]
        print(
            f"{r['dataset']:18s} {r['query_count']:>8d} "
            f"{m['hit@10']:>8.4f} {m['recall@10']:>8.4f} "
            f"{m['ndcg@10']:>8.4f} {m['mrr@10']:>8.4f}"
        )
    print("-" * 90)
    print(
        f"{'MACRO':18s} {'':>8s} "
        f"{macro_metrics['hit@10']:>8.4f} "
        f"{macro_metrics['recall@10']:>8.4f} "
        f"{macro_metrics['ndcg@10']:>8.4f} "
        f"{macro_metrics['mrr@10']:>8.4f}"
    )
    print()
    print(
        f"M7 COMPLETE — locked test finished in "
        f"{manifest['wall_clock_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())