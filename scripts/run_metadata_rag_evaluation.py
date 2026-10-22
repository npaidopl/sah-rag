# ruff: noqa: E402
"""Metadata-RAG L6 - Evaluation over the 4-dataset scope.

For each dataset:

  1. Load retrieval-v1/results.jsonl (ranked canonical doc lists).
  2. Load canonical ground truth from the dataset's examples.jsonl.
  3. Compute per-query metrics at cutoffs {1, 5, 10, 20, 50, 100}:
       Hit@k, Recall@k, Precision@k, MRR@k, nDCG@k
  4. Aggregate per dataset with macro averaging.
  5. Write evaluation-v1/: query_metrics.jsonl, aggregate_metrics.json,
     manifest.json, checksums.json.

Cross-dataset macro aggregation is done by a separate comparison
script (Cell 44).

This script is deterministic, retrieval-only, and never invokes any
generation, LLM judge, or answer-quality model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sah_rag.canonical.loaders import iter_canonical_examples

DEFAULT_OUTPUT_ROOT: Final[Path] = (
    REPO_ROOT / "outputs" / "external_metadata_rag"
)
EXPECTED_DATASETS: Final[tuple[str, ...]] = (
    "clerk",
    "cuad",
    "greek_bar_bench",
    "lara",
)

CUTOFFS: Final[tuple[int, ...]] = (1, 5, 10, 20, 50, 100)

RETRIEVAL_DIR: Final[str] = "retrieval-v1"
EVALUATION_DIR: Final[str] = "evaluation-v1"

RESULTS_FILE: Final[str] = "results.jsonl"
OUTPUT_QUERY_METRICS: Final[str] = "query_metrics.jsonl"
OUTPUT_AGGREGATE: Final[str] = "aggregate_metrics.json"
OUTPUT_MANIFEST: Final[str] = "manifest.json"
OUTPUT_CHECKSUMS: Final[str] = "checksums.json"

EVALUATION_ARTIFACT_TYPE: Final[str] = "metadata_rag_evaluation"
EVALUATION_ARTIFACT_VERSION: Final[str] = "1.0"

PRIMARY_METRIC: Final[str] = "macro_document_recall_at_10"


class MetadataRAGEvaluationError(ValueError):
    """Raised when evaluation violates its contract."""


# -------------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)


def _write_json(path: Path, obj: Any) -> None:
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: Sequence[Any]) -> int:
    if not rows:
        _atomic_write_bytes(path, b"\n")
        return 0
    parts: list[str] = []
    for row in rows:
        payload = (
            row.model_dump(mode="json")
            if hasattr(row, "model_dump")
            else row
        )
        parts.append(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
    _atomic_write_bytes(path, ("\n".join(parts) + "\n").encode("utf-8"))
    return len(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise MetadataRAGEvaluationError(f"missing file: {path}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise MetadataRAGEvaluationError(
                f"{path.name}:{lineno} not a JSON object"
            )
        rows.append(parsed)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MetadataRAGEvaluationError(f"missing file: {path}")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise MetadataRAGEvaluationError(
            f"{path.name} is not a JSON object"
        )
    return parsed


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------


def _hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for doc_id in ranked[:k]:
        if doc_id in relevant:
            return 1.0
    return 0.0


def _recall_at_k(
    ranked: Sequence[str], relevant: set[str], k: int
) -> float:
    if not relevant:
        raise MetadataRAGEvaluationError(
            "recall_at_k requires at least one relevant document"
        )
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def _precision_at_k(
    ranked: Sequence[str], relevant: set[str], k: int
) -> float:
    if k <= 0:
        raise MetadataRAGEvaluationError("k must be positive")
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / k


def _reciprocal_rank(
    ranked: Sequence[str], relevant: set[str], k: int
) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(
    ranked: Sequence[str], relevant: set[str], k: int
) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        rel = 1.0 if doc_id in relevant else 0.0
        gain = (2.0**rel) - 1.0
        discount = 1.0 / math.log2(rank + 1)
        dcg += gain * discount
    return dcg


def _ndcg_at_k(
    ranked: Sequence[str], relevant: set[str], k: int
) -> float:
    if not relevant:
        raise MetadataRAGEvaluationError(
            "ndcg_at_k requires at least one relevant document"
        )
    dcg = _dcg_at_k(ranked, relevant, k)
    ideal_count = min(k, len(relevant))
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    if idcg <= 0.0:
        raise MetadataRAGEvaluationError(
            "ideal DCG is zero; cannot compute nDCG"
        )
    return dcg / idcg


def _compute_query_metrics(
    *,
    query_id: str,
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: set[str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "query_id": query_id,
        "ranked_count": len(ranked_doc_ids),
        "relevant_count": len(relevant_doc_ids),
        "metrics": {},
    }
    metrics: dict[str, float] = {}
    for k in CUTOFFS:
        metrics[f"hit_at_{k}"] = _hit_at_k(
            ranked_doc_ids, relevant_doc_ids, k
        )
        metrics[f"precision_at_{k}"] = _precision_at_k(
            ranked_doc_ids, relevant_doc_ids, k
        )
        metrics[f"recall_at_{k}"] = _recall_at_k(
            ranked_doc_ids, relevant_doc_ids, k
        )
        metrics[f"mrr_at_{k}"] = _reciprocal_rank(
            ranked_doc_ids, relevant_doc_ids, k
        )
        metrics[f"ndcg_at_{k}"] = _ndcg_at_k(
            ranked_doc_ids, relevant_doc_ids, k
        )
    entry["metrics"] = metrics
    return entry


def _aggregate_macro(
    per_query: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if not per_query:
        raise MetadataRAGEvaluationError(
            "cannot aggregate an empty set of queries"
        )
    n = len(per_query)
    sums: dict[str, float] = {}
    for entry in per_query:
        for key, value in entry["metrics"].items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {
        "query_count": n,
        "mean": {k: v / n for k, v in sums.items()},
    }


# -------------------------------------------------------------------------
# Per-dataset evaluation
# -------------------------------------------------------------------------


def _load_ground_truth(
    dataset: str,
) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for example in iter_canonical_examples(dataset):
        qid = example.query.query_id
        if qid in out:
            raise MetadataRAGEvaluationError(
                f"duplicate query_id in ground truth: {qid}"
            )
        out[qid] = tuple(example.ground_truth.relevant_ids)
    return out


def _load_retrieval_results(
    dataset_dir: Path,
) -> dict[str, list[str]]:
    """Load per-query ranked canonical doc lists with first-occurrence
    deduplication.

    Retrieval emits one RankedItem per window. A single canonical
    document may therefore appear multiple times within one query's
    ranked list. Evaluation is at the canonical-document level, so
    duplicates must be removed by first-occurrence rule, exactly as
    the document expansion stage does. Without this deduplication the
    metrics would count the same document multiple times and recall
    could exceed 1.0.
    """
    rows = _read_jsonl(
        dataset_dir / RETRIEVAL_DIR / RESULTS_FILE
    )
    out: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row.get("query_id", ""))
        if not qid:
            raise MetadataRAGEvaluationError(
                "retrieval row missing query_id"
            )
        if qid in out:
            raise MetadataRAGEvaluationError(
                f"duplicate retrieval query_id: {qid}"
            )
        ranked_items = row.get("ranked_items", [])
        seen: set[str] = set()
        ranked_doc_ids: list[str] = []
        for item in ranked_items:
            doc_id = item.get("canonical_doc_id")
            if not doc_id:
                raise MetadataRAGEvaluationError(
                    "ranked item missing canonical_doc_id"
                )
            doc_id_str = str(doc_id)
            if doc_id_str in seen:
                continue
            seen.add(doc_id_str)
            ranked_doc_ids.append(doc_id_str)
        out[qid] = ranked_doc_ids
    return out


def _evaluate_one(
    *,
    dataset: str,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    dataset_dir = output_root / dataset
    if not dataset_dir.is_dir():
        raise MetadataRAGEvaluationError(
            f"dataset dir not found: {dataset_dir}"
        )
    if force and (dataset_dir / EVALUATION_DIR).exists():
        import shutil

        shutil.rmtree(dataset_dir / EVALUATION_DIR)

    t0 = time.monotonic()

    print(f"[1/4] load retrieval results: {dataset}")
    ranked_by_query = _load_retrieval_results(dataset_dir)
    print(f"      queries: {len(ranked_by_query)}")

    print("[2/4] load ground truth")
    gt_by_query = _load_ground_truth(dataset)
    print(f"      ground truth queries: {len(gt_by_query)}")

    # Bijection check: every retrieval query has GT and vice versa.
    only_retrieval = set(ranked_by_query) - set(gt_by_query)
    only_gt = set(gt_by_query) - set(ranked_by_query)
    if only_retrieval or only_gt:
        raise MetadataRAGEvaluationError(
            "retrieval <-> ground truth mismatch: "
            f"only_retrieval={sorted(only_retrieval)[:5]} "
            f"only_gt={sorted(only_gt)[:5]}"
        )

    print("[3/4] compute metrics")
    per_query: list[dict[str, Any]] = []
    excluded_zero_relevance: list[str] = []
    for qid in sorted(ranked_by_query):
        relevant = set(gt_by_query[qid])
        if not relevant:
            excluded_zero_relevance.append(qid)
            continue
        per_query.append(
            _compute_query_metrics(
                query_id=qid,
                ranked_doc_ids=ranked_by_query[qid],
                relevant_doc_ids=relevant,
            )
        )

    if not per_query:
        raise MetadataRAGEvaluationError(
            f"{dataset}: all queries excluded (no relevant docs)"
        )

    aggregate = _aggregate_macro(per_query)

    print("[4/4] write outputs")
    evaluation_dir = dataset_dir / EVALUATION_DIR
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    n = _write_jsonl(
        evaluation_dir / OUTPUT_QUERY_METRICS, per_query
    )

    aggregate_with_context = {
        "dataset_id": dataset,
        "query_count": aggregate["query_count"],
        "excluded_zero_relevance_count": len(
            excluded_zero_relevance
        ),
        "cutoffs": list(CUTOFFS),
        "primary_metric": PRIMARY_METRIC,
        "mean": aggregate["mean"],
    }
    _write_json(
        evaluation_dir / OUTPUT_AGGREGATE, aggregate_with_context
    )

    elapsed = time.monotonic() - t0
    manifest = {
        "artifact_type": EVALUATION_ARTIFACT_TYPE,
        "artifact_version": EVALUATION_ARTIFACT_VERSION,
        "dataset_id": dataset,
        "query_count": n,
        "excluded_zero_relevance_count": len(
            excluded_zero_relevance
        ),
        "cutoffs": list(CUTOFFS),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_value": aggregate["mean"].get(
            "recall_at_10"
        ),
        "generation_enabled": False,
        "llm_judge_enabled": False,
        "source_results_sha256": _sha256_file(
            dataset_dir / RETRIEVAL_DIR / RESULTS_FILE
        ),
        "elapsed_seconds": round(elapsed, 3),
    }
    _write_json(evaluation_dir / OUTPUT_MANIFEST, manifest)

    checksums = {
        OUTPUT_QUERY_METRICS: _sha256_file(
            evaluation_dir / OUTPUT_QUERY_METRICS
        ),
        OUTPUT_AGGREGATE: _sha256_file(
            evaluation_dir / OUTPUT_AGGREGATE
        ),
        OUTPUT_MANIFEST: _sha256_file(
            evaluation_dir / OUTPUT_MANIFEST
        ),
    }
    _write_json(evaluation_dir / OUTPUT_CHECKSUMS, checksums)

    return {
        "dataset": dataset,
        "query_count": n,
        "excluded_zero_relevance": len(excluded_zero_relevance),
        "primary_metric_value": aggregate["mean"].get(
            "recall_at_10"
        ),
        "elapsed_seconds": round(elapsed, 3),
    }


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Metadata-RAG L6 evaluation over the four datasets."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(EXPECTED_DATASETS),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        if dataset not in EXPECTED_DATASETS:
            raise SystemExit(f"unknown dataset: {dataset}")
        summary = _evaluate_one(
            dataset=dataset,
            output_root=args.output_root,
            force=args.force,
        )
        summaries.append(summary)

    print("[OK] Metadata-RAG evaluation")
    print(
        f"  {'dataset':<18} {'queries':>8} {'excluded':>9} "
        f"{'recall@10':>10}"
    )
    for s in summaries:
        value = s["primary_metric_value"]
        value_str = f"{value:.4f}" if value is not None else "n/a"
        print(
            f"  {s['dataset']:<18} {s['query_count']:>8} "
            f"{s['excluded_zero_relevance']:>9} {value_str:>10}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
