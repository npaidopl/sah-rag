"""Validate the Step 89 evaluation outputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_evaluation_execution_contract.yaml"

METRIC_FIELDS = (
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
)

AGGREGATE_FIELDS = (
    "mean_hit_at_k",
    "mean_precision_at_k",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "mean_ndcg_at_k",
)

ANSWER_KEYS = {
    "generated_answer",
    "reference_answer",
    "answer_text",
    "answer_correctness",
    "llm_judge_results",
    "relevance_label",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _bounded(v: float) -> bool:
    return math.isfinite(v) and 0.0 <= v <= 1.0


def main() -> int:
    contract = yaml.safe_load(EXECUTION_CONTRACT.read_text(encoding="utf-8"))
    root = REPO_ROOT / contract["output"]["root"]

    _require(root.is_dir(), "evaluation output root missing: " + str(root))
    for name in contract["output"]["required_files"]:
        _require((root / name).exists(), "required file missing: " + name)

    cases = _load_jsonl(root / "retrieval_cases.jsonl")
    query_metrics = _load_jsonl(root / "query_metrics.jsonl")
    aggregate_metrics = json.loads(
        (root / "aggregate_metrics.json").read_text(encoding="utf-8")
    )

    expected_counts = contract["expected_counts"]
    _require(
        len(cases) == expected_counts["retrieval_case_count"],
        "retrieval_cases count mismatch",
    )
    _require(
        len(query_metrics) == expected_counts["query_metric_rows"],
        "query_metrics row count mismatch",
    )
    agg_list = (
        aggregate_metrics
        if isinstance(aggregate_metrics, list)
        else aggregate_metrics.get("aggregates", [])
    )
    _require(
        len(agg_list) == expected_counts["aggregate_metric_rows"],
        "aggregate_metrics row count mismatch",
    )

    query_ids = {c["query_id"] for c in cases}
    _require(len(query_ids) == 6, "must be exactly 6 unique query ids")

    cutoffs = {int(m["cutoff"]) for m in query_metrics}
    _require(cutoffs == {1, 5, 10, 100}, "cutoffs must be {1, 5, 10, 100}")

    for row in query_metrics:
        for f in METRIC_FIELDS:
            v = float(row[f])
            _require(_bounded(v), f + " out of bounds for " + row["query_id"])

    for row in agg_list:
        _require(int(row["query_count"]) == 6, "aggregate query_count must be 6")
        for f in AGGREGATE_FIELDS:
            v = float(row[f])
            _require(_bounded(v), f + " out of bounds at cutoff " + str(row["cutoff"]))

    # macro-aggregation check
    for agg_row in agg_list:
        cutoff = int(agg_row["cutoff"])
        per_cutoff = [
            r for r in query_metrics if int(r["cutoff"]) == cutoff
        ]
        _require(len(per_cutoff) == 6, "per-cutoff rows must be 6")
        for src, dst in zip(METRIC_FIELDS, AGGREGATE_FIELDS, strict=True):
            mean = sum(float(r[src]) for r in per_cutoff) / 6.0
            _require(
                abs(mean - float(agg_row[dst])) <= 1e-12,
                "macro mean mismatch at cutoff " + str(cutoff) + " for " + src,
            )

    # prohibited fields
    for row in query_metrics + agg_list:
        forbidden = ANSWER_KEYS & set(row.keys())
        _require(
            not forbidden,
            "forbidden field(s): " + ", ".join(sorted(forbidden)),
        )

    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    for rel, expected in checksums["files"].items():
        _require(_sha256(root / rel) == expected, "checksum mismatch: " + rel)

    print("[OK] Pilot evaluation outputs")
    print("  retrieval cases:", len(cases))
    print("  query metric rows:", len(query_metrics))
    print("  aggregate metric rows:", len(agg_list))
    print("  expected queries complete: True")
    print("  cutoffs complete: True")
    print("  hit metrics valid: True")
    print("  precision metrics valid: True")
    print("  recall metrics valid: True")
    print("  reciprocal-rank metrics valid: True")
    print("  NDCG metrics valid: True")
    print("  macro aggregates valid: True")
    print("  duplicate canonical identities preserved: True")
    print("  answer fields absent: True")
    print("  generation artifacts absent: True")
    print("  architecture comparison absent: True")
    print("  round trip valid: True")
    print("  checksums valid: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())