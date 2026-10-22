# ruff: noqa: E402
"""LongRAG L6 - Ground-truth evaluation.

Separate script from the retriever. Reads:
  - outputs/external_longrag/<ds>/retrieval-v1/results.jsonl
  - ${SAH_RAG_CANONICAL_ROOT}/<ds>/examples.jsonl

Ground truth is loaded ONLY here, never in the retriever.

Metrics (macro per dataset):
  Hit@K, Recall@K, Precision@10, MRR@K, nDCG@K
for K in {1, 5, 10, 20, 50} (Hit/Recall/MRR), 10 only (Precision),
and {10, 50} for nDCG.

Zero-relevance queries are excluded from all metrics and counted
explicitly per common_retrieval_evaluation_contract.yaml.
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
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sah_rag.canonical.loaders import iter_canonical_examples

OUTPUT_ROOT = REPO_ROOT / "outputs" / "external_longrag"
REPORTS_ROOT = REPO_ROOT / "reports"

EXPECTED_DATASETS: tuple[str, ...] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

CUTOFFS: tuple[int, ...] = (1, 5, 10, 20, 50)
PRECISION_K = 10
NDCG_CUTOFFS: tuple[int, ...] = (10, 50)


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


def load_retrieval_results(dataset_id: str) -> dict[str, list[str]]:
    """Return {query_id: [ranked canonical doc_ids]}."""
    path = OUTPUT_ROOT / dataset_id / "retrieval-v1" / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            items = row["ranked_items"]
            # Ensure rank-ordered (should already be, but be safe)
            items_sorted = sorted(items, key=lambda x: x["rank"])
            out[qid] = [it["canonical_doc_id"] for it in items_sorted]
    return out


def load_ground_truth(dataset_id: str) -> dict[str, list[str]]:
    """Return {query_id: [relevant doc_ids]}."""
    out: dict[str, list[str]] = {}
    for ex in iter_canonical_examples(dataset_id):
        qid = ex.query.query_id
        rel = list(ex.ground_truth.relevant_ids)
        out[qid] = rel
    return out


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / k


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        rel = 1.0 if d in relevant else 0.0
        gain = (2.0 ** rel) - 1.0
        dcg += gain / math.log2(i + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(
        1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_dataset(dataset_id: str) -> dict[str, Any]:
    t_start = __import__("time").perf_counter()
    print(f"[{dataset_id}] loading retrieval + ground truth ...")
    ranked_by_qid = load_retrieval_results(dataset_id)
    gt_by_qid = load_ground_truth(dataset_id)

    all_qids = set(gt_by_qid.keys())
    missing_results = all_qids - set(ranked_by_qid.keys())
    extra_results = set(ranked_by_qid.keys()) - all_qids
    if extra_results:
        raise RuntimeError(
            f"{dataset_id}: results contain qids not in examples: "
            f"{list(extra_results)[:3]}"
        )
    if missing_results:
        raise RuntimeError(
            f"{dataset_id}: {len(missing_results)} queries without results: "
            f"{list(missing_results)[:3]}"
        )

    per_query: list[dict[str, Any]] = []
    excluded_zero_rel = 0

    for qid in sorted(all_qids):
        ranked = ranked_by_qid[qid]
        relevant = gt_by_qid[qid]

        if len(relevant) == 0:
            excluded_zero_rel += 1
            continue

        relevant_set = set(relevant)

        row: dict[str, Any] = {
            "query_id": qid,
            "num_relevant": len(relevant),
            "num_retrieved": len(ranked),
        }
        for k in CUTOFFS:
            row[f"hit@{k}"] = hit_at_k(ranked, relevant_set, k)
            row[f"recall@{k}"] = recall_at_k(ranked, relevant_set, k)
        row[f"precision@{PRECISION_K}"] = precision_at_k(
            ranked, relevant_set, PRECISION_K
        )
        for k in CUTOFFS:
            row[f"mrr@{k}"] = reciprocal_rank(ranked, relevant_set, k)
        for k in NDCG_CUTOFFS:
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant_set, k)

        per_query.append(row)

    n_eval = len(per_query)

    aggregates: dict[str, float] = {}
    if n_eval > 0:
        keys_to_average = [
            f"hit@{k}" for k in CUTOFFS
        ] + [
            f"recall@{k}" for k in CUTOFFS
        ] + [
            f"precision@{PRECISION_K}",
        ] + [
            f"mrr@{k}" for k in CUTOFFS
        ] + [
            f"ndcg@{k}" for k in NDCG_CUTOFFS
        ]
        for key in keys_to_average:
            aggregates[key] = sum(
                r[key] for r in per_query
            ) / n_eval

    stage = OUTPUT_ROOT / dataset_id / "evaluation-v1"
    stage.mkdir(parents=True, exist_ok=True)

    per_query_path = stage / "per_query_metrics.jsonl"
    write_jsonl_atomic(per_query_path, per_query)

    report: dict[str, Any] = {
        "artifact_type": "longrag_full_track_evaluation_report",
        "artifact_version": "1.0",
        "dataset_id": dataset_id,
        "query_count_total": len(all_qids),
        "query_count_evaluated": n_eval,
        "queries_excluded_zero_relevance": excluded_zero_rel,
        "cutoffs": list(CUTOFFS),
        "precision_k": PRECISION_K,
        "ndcg_cutoffs": list(NDCG_CUTOFFS),
        "macro_metrics": aggregates,
        "per_query_sha256": sha256_file(per_query_path),
        "wall_clock_seconds": round(
            __import__("time").perf_counter() - t_start, 3
        ),
    }
    write_json_atomic(stage / "per_dataset_metrics.json", report)

    print(
        f"[{dataset_id}] evaluated {n_eval} / {len(all_qids)} "
        f"(excluded {excluded_zero_rel} zero-relevance)"
    )
    return report


def run(*, datasets: tuple[str, ...]) -> int:
    reports: list[dict[str, Any]] = []
    for ds in datasets:
        try:
            reports.append(evaluate_dataset(ds))
        except Exception as exc:
            print(f"[{ds}] FAILED: {exc}")
            reports.append({
                "dataset_id": ds, "status": "FAILED", "error": str(exc),
            })

    # Macro-average across datasets (equal weight per dataset)
    eval_reports = [r for r in reports if "macro_metrics" in r]
    macro: dict[str, float] = {}
    if eval_reports:
        all_keys = set(eval_reports[0]["macro_metrics"].keys())
        for key in sorted(all_keys):
            macro[key] = sum(
                r["macro_metrics"][key] for r in eval_reports
            ) / len(eval_reports)

    cross = {
        "artifact_type": "longrag_full_track_cross_dataset_evaluation",
        "artifact_version": "1.0",
        "execution_id": "longrag-l6-evaluation-v1",
        "status": "SUCCEEDED"
        if all("macro_metrics" in r for r in reports)
        else "PARTIAL",
        "datasets": [
            {
                "dataset_id": r["dataset_id"],
                "query_count_total": r.get("query_count_total"),
                "query_count_evaluated": r.get("query_count_evaluated"),
                "queries_excluded_zero_relevance": r.get(
                    "queries_excluded_zero_relevance"
                ),
                "macro_metrics": r.get("macro_metrics"),
            }
            for r in reports
        ],
        "macro_average_across_datasets": macro,
        "primary_metric": "macro_document_Recall_at_10",
        "primary_metric_value": macro.get("recall@10"),
    }
    write_json_atomic(OUTPUT_ROOT / "_l6_cross_dataset_evaluation.json", cross)

    print()
    print("L6 CROSS-DATASET RESULTS")
    print("=" * 78)
    header = f"{'dataset':18s} {'evaluated':>10s} {'excl':>6s} " + \
             " ".join(f"{'H@'+str(k):>6s}" for k in (1, 5, 10, 50))
    print(header)
    for r in reports:
        if "macro_metrics" not in r:
            print(f"  {r['dataset_id']:18s}  FAILED: {r.get('error')}")
            continue
        m = r["macro_metrics"]
        line = f"  {r['dataset_id']:18s} {r['query_count_evaluated']:>10d} " + \
               f"{r['queries_excluded_zero_relevance']:>6d} " + \
               " ".join(f"{m['hit@'+str(k)]:>6.4f}" for k in (1, 5, 10, 50))
        print(line)
    print()
    print("RECALL")
    for r in eval_reports:
        m = r["macro_metrics"]
        line = f"  {r['dataset_id']:18s} " + \
               " ".join(f"{m['recall@'+str(k)]:>6.4f}" for k in (1, 5, 10, 50))
        print(line)
    print()
    print("nDCG / MRR@10 / Precision@10")
    for r in eval_reports:
        m = r["macro_metrics"]
        print(
            f"  {r['dataset_id']:18s} "
            f"nDCG@10={m['ndcg@10']:.4f}  "
            f"nDCG@50={m['ndcg@50']:.4f}  "
            f"MRR@10={m['mrr@10']:.4f}  "
            f"P@10={m['precision@10']:.4f}"
        )
    print()
    print("MACRO AVERAGE ACROSS DATASETS")
    for k in sorted(macro.keys()):
        print(f"  {k:20s}  {macro[k]:.4f}")

    return 0 if all("macro_metrics" in r for r in reports) else 1


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