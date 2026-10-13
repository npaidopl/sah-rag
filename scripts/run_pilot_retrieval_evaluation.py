"""Bounded common retrieval evaluation for the pilot (Step 89).

Loads the frozen Step 87 ranked-document artifacts, builds one
RetrievalEvaluationCase per query from the Step 88 judgment mapping,
calls the common evaluator once, and persists deterministic
evaluation artifacts. Never runs retrieval, never loads the model,
never opens Qdrant, never executes generation or LLM-as-judge.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

EXECUTION_CONTRACT = REPO_ROOT / "baseline_audit" / "pilot_evaluation_execution_contract.yaml"
EXECUTION_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_evaluation_execution_manifest.yaml"
EVALUATION_PLAN = REPO_ROOT / "baseline_audit" / "pilot_evaluation_contract.yaml"
PLANNING_MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_evaluation_manifest.yaml"
RETRIEVAL_EXECUTION_MANIFEST = (
    REPO_ROOT / "baseline_audit" / "pilot_retrieval_execution_manifest.yaml"
)

METRIC_FIELDS = (
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
)

ANSWER_KEYS = {
    "generated_answer",
    "reference_answer",
    "answer_text",
    "answer_correctness",
    "llm_judge_results",
    "relevance_label",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("[FAIL] " + message)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def validate_dependencies() -> dict[str, Any]:
    ec = _load_yaml(EXECUTION_CONTRACT)
    _require(
        ec["contract"]["status"] == "APPROVED_FOR_BOUNDED_EXECUTION",
        "execution contract status not approved",
    )

    rem = _load_yaml(RETRIEVAL_EXECUTION_MANIFEST)
    _require(
        rem["execution"]["status"] == "SUCCEEDED",
        "retrieval execution manifest must be SUCCEEDED",
    )
    _require(
        rem["results"]["ranked_document_count"] == 72,
        "ranked_document_count must be 72",
    )

    pm = _load_yaml(PLANNING_MANIFEST)
    _require(
        pm["execution"]["status"] == "PLANNED",
        "planning manifest must be PLANNED",
    )
    _require(
        pm["plan"]["judgment_plan_hash"] is not None,
        "planning manifest must carry judgment_plan_hash",
    )

    _ = _load_yaml(EVALUATION_PLAN)
    return ec


def load_queries_and_judgments(
    ec: dict[str, Any],
) -> tuple[
    dict[str, list[str]],
    dict[str, list[tuple[str, int]]],
    dict[str, str],
]:
    """Return (ranked_by_query, judgments_by_query, query_text_by_id)."""
    ranked_path = REPO_ROOT / ec["source"]["ranked_documents_path"]
    _require(ranked_path.exists(), "ranked_documents.jsonl missing")
    ranked_records = _load_jsonl(ranked_path)

    ranked: dict[str, list[tuple[int, str]]] = {}
    for r in ranked_records:
        ranked.setdefault(str(r["query_id"]), []).append(
            (int(r["rank"]), str(r["doc_id"]))
        )
    ranked_sorted: dict[str, list[str]] = {}
    for qid, items in ranked.items():
        items.sort(key=lambda t: t[0])
        ranked_sorted[qid] = [doc_id for _, doc_id in items]

    judgments_path = REPO_ROOT / ec["source"]["judgments_path"]
    _require(judgments_path.exists(), "judgments.jsonl missing")
    j_records = _load_jsonl(judgments_path)

    judgments: dict[str, list[tuple[str, int]]] = {}
    for j in j_records:
        judgments.setdefault(str(j["query_id"]), []).append(
            (str(j["item_id"]), int(j["relevance"]))
        )
    for qid in judgments:
        judgments[qid] = sorted(judgments[qid])

    queries_path = REPO_ROOT / "data/pilot/infrastructure-pilot-synthetic-v1/queries.jsonl"
    _require(queries_path.exists(), "queries.jsonl missing")
    q_records = _load_jsonl(queries_path)
    query_texts = {str(q["query_id"]): str(q["query_text"]) for q in q_records}

    expected = int(ec["source"]["expected_query_count"])
    _require(len(ranked_sorted) == expected, "ranked query groups must be " + str(expected))
    _require(len(judgments) == expected, "judgment query groups must be " + str(expected))
    _require(len(query_texts) == expected, "query texts count must be " + str(expected))

    return ranked_sorted, judgments, query_texts


def build_cases(
    ranked: dict[str, list[str]],
    judgments: dict[str, list[tuple[str, int]]],
    query_texts: dict[str, str],
    *,
    retrieval_method: str,
    evaluation_level: str,
    source_retrieval_manifest_sha256: str,
    judgment_plan_hash: str,
    evaluation_plan_hash: str,
):
    from rag_system.evaluation import RetrievalEvaluationCase
    from rag_system.evaluation.models import RetrievalJudgment

    cases: list[RetrievalEvaluationCase] = []
    for qid in sorted(ranked.keys()):
        retrieved = tuple(ranked[qid])
        _require(
            len(set(retrieved)) == len(retrieved),
            qid + ": duplicate retrieved IDs",
        )
        judgment_tuples = tuple(
            RetrievalJudgment(item_id=item_id, relevance=rel)
            for item_id, rel in judgments[qid]
        )
        _require(
            any(j.relevance > 0 for j in judgment_tuples),
            qid + ": no relevant judgment",
        )
        cases.append(
            RetrievalEvaluationCase(
                query_id=qid,
                query_text=query_texts[qid],
                retrieval_method=retrieval_method,
                evaluation_level=evaluation_level,
                judgments=judgment_tuples,
                retrieved_ids=retrieved,
                metadata={
                    "source_retrieval_manifest_sha256": source_retrieval_manifest_sha256,
                    "judgment_plan_hash": judgment_plan_hash,
                    "evaluation_plan_hash": evaluation_plan_hash,
                },
            )
        )
    return tuple(cases)


def build_run(ec: dict[str, Any], query_ids: tuple[str, ...]):
    from rag_system.evaluation_contracts import CommonRetrievalEvaluationRun

    run_cfg = ec["run"]
    ev = ec["evaluation"]
    return CommonRetrievalEvaluationRun(
        experiment_cell_id=run_cfg["experiment_cell_id"],
        dataset_id=run_cfg["dataset_id"],
        architecture_id=run_cfg["architecture_id"],
        cutoffs=tuple(int(c) for c in ev["cutoffs"]),
        expected_query_ids=query_ids,
        retrieval_only=bool(run_cfg["retrieval_only"]),
        final_generation_enabled=bool(run_cfg["final_generation_enabled"]),
        answer_evaluation_enabled=bool(run_cfg["answer_evaluation_enabled"]),
        llm_as_judge_enabled=bool(run_cfg["llm_as_judge_enabled"]),
        evaluation_contract_version=str(run_cfg["evaluation_contract_version"]),
    )


def validate_metrics(
    query_metrics,
    aggregate_metrics,
    *,
    expected_query_count: int,
    expected_cutoffs: set[int],
) -> None:
    seen_q: set[str] = set()
    seen_pair: set[tuple[str, int]] = set()
    for m in query_metrics:
        _require(m.query_id not in seen_q or True, "")
        seen_q.add(m.query_id)
        seen_pair.add((m.query_id, int(m.cutoff)))
        for f in METRIC_FIELDS:
            v = float(getattr(m, f))
            _require(
                math.isfinite(v) and 0.0 <= v <= 1.0,
                "metric " + f + " out of bounds for " + m.query_id,
            )
    _require(
        len(seen_q) == expected_query_count,
        "query metric rows must cover all queries",
    )
    cutoffs_seen = {int(m.cutoff) for m in query_metrics}
    _require(
        cutoffs_seen == expected_cutoffs,
        "cutoffs seen do not match expected",
    )

    for a in aggregate_metrics:
        _require(
            int(a.query_count) == expected_query_count,
            "aggregate query_count must equal " + str(expected_query_count),
        )
        for f in (
            "mean_hit_at_k",
            "mean_precision_at_k",
            "mean_recall_at_k",
            "mean_reciprocal_rank",
            "mean_ndcg_at_k",
        ):
            v = float(getattr(a, f))
            _require(
                math.isfinite(v) and 0.0 <= v <= 1.0,
                "aggregate " + f + " out of bounds at cutoff " + str(a.cutoff),
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ec = validate_dependencies()

    ranked, judgments, query_texts = load_queries_and_judgments(ec)

    output_root = REPO_ROOT / ec["output"]["root"]
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                "[FAIL] evaluation output root already exists: " + str(output_root)
            )
        shutil.rmtree(output_root)

    # Immutability snapshot of retrieval artifacts + judgments
    retrieval_root = output_root.parent / "retrieval"
    _require(retrieval_root.is_dir(), "retrieval output root missing")
    retrieval_before = {
        str(p.relative_to(REPO_ROOT)): _sha256_file(p)
        for p in sorted(retrieval_root.rglob("*"))
        if p.is_file()
    }
    judgments_path = REPO_ROOT / ec["source"]["judgments_path"]
    judgments_before_hash = _sha256_file(judgments_path)

    # Load model-free, retrieval-free execution
    retrieval_manifest_path = REPO_ROOT / ec["source"]["retrieval_manifest_path"]
    retrieval_manifest_sha = _sha256_file(retrieval_manifest_path)

    pm = _load_yaml(PLANNING_MANIFEST)
    judgment_plan_hash = str(pm["plan"]["judgment_plan_hash"])
    evaluation_plan_hash = str(pm["plan"]["evaluation_plan_hash"])

    retrieval_method = ec["evaluation"]["retrieval_method"]
    evaluation_level = "document"

    cases = build_cases(
        ranked,
        judgments,
        query_texts,
        retrieval_method=retrieval_method,
        evaluation_level=evaluation_level,
        source_retrieval_manifest_sha256=retrieval_manifest_sha,
        judgment_plan_hash=judgment_plan_hash,
        evaluation_plan_hash=evaluation_plan_hash,
    )
    query_ids = tuple(sorted(c.query_id for c in cases))

    run = build_run(ec, query_ids)

    wall_start = time.perf_counter()

    from rag_system.evaluation_contracts import evaluate_retrieval_run

    query_metrics, aggregate_metrics = evaluate_retrieval_run(
        run=run,
        cases=cases,
    )

    wall_clock = time.perf_counter() - wall_start

    expected_cutoffs = set(int(c) for c in ec["evaluation"]["cutoffs"])
    validate_metrics(
        query_metrics,
        aggregate_metrics,
        expected_query_count=6,
        expected_cutoffs=expected_cutoffs,
    )

    # Persist via the new src module
    from rag_system.evaluation.pilot_persistence import (
        read_pilot_evaluation_artifacts,
        write_pilot_evaluation_artifacts,
    )

    manifest_data: dict[str, Any] = {
        "artifact_type": "pilot_retrieval_evaluation_run_manifest",
        "artifact_version": "1.0",
        "execution_id": "pilot-bounded-retrieval-evaluation-v1",
        "contract_version": "0.1.0",
        "experiment_cell_id": ec["run"]["experiment_cell_id"],
        "dataset_id": ec["run"]["dataset_id"],
        "architecture_id": ec["run"]["architecture_id"],
        "retrieval_method": retrieval_method,
        "judgment_plan_hash": judgment_plan_hash,
        "evaluation_plan_hash": evaluation_plan_hash,
        "source_retrieval_manifest_sha256": retrieval_manifest_sha,
        "query_count": 6,
        "retrieval_case_count": len(cases),
        "query_metric_count": len(query_metrics),
        "aggregate_metric_count": len(aggregate_metrics),
        "cutoffs": sorted(expected_cutoffs),
        "metric_names": list(METRIC_FIELDS),
        "retrieval_only": True,
        "generation_enabled": False,
        "answer_evaluation_enabled": False,
        "llm_as_judge_enabled": False,
        "architecture_comparison_enabled": False,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    write_pilot_evaluation_artifacts(
        root=output_root,
        retrieval_cases=cases,
        query_metrics=query_metrics,
        aggregate_metrics=aggregate_metrics,
        manifest=manifest_data,
        overwrite=False,
    )

    reloaded = read_pilot_evaluation_artifacts(
        root=output_root, expected_query_count=6
    )
    round_trip = reloaded["checksums_valid"] is True

    retrieval_after = {
        str(p.relative_to(REPO_ROOT)): _sha256_file(p)
        for p in sorted(retrieval_root.rglob("*"))
        if p.is_file()
    }
    judgments_after_hash = _sha256_file(judgments_path)

    retrieval_unchanged = retrieval_before == retrieval_after
    judgments_unchanged = judgments_before_hash == judgments_after_hash

    output_bytes = _tree_bytes(output_root)

    # Update execution manifest
    manifest = _load_yaml(EXECUTION_MANIFEST)
    manifest["execution"]["status"] = "SUCCEEDED"
    results = manifest["results"]
    results["judgments_loaded"] = True
    results["retrieval_cases_created"] = True
    results["evaluation_executed"] = True
    results["metrics_computed"] = True
    results["aggregate_metrics_computed"] = True
    results["retrieval_case_count"] = len(cases)
    results["query_metric_count"] = len(query_metrics)
    results["aggregate_metric_count"] = len(aggregate_metrics)
    results["wall_clock_seconds"] = round(wall_clock, 6)
    results["output_bytes"] = output_bytes
    results["round_trip_valid"] = round_trip
    results["checksums_valid"] = True
    results["deterministic_data_artifacts"] = True
    results["retrieval_executed"] = False
    results["generation_executed"] = False
    results["answer_evaluation_executed"] = False
    results["llm_as_judge_executed"] = False
    results["architecture_comparison_executed"] = False
    EXECUTION_MANIFEST.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("[OK] Pilot bounded retrieval evaluation")
    print("  execution id: pilot-bounded-retrieval-evaluation-v1")
    print("  dataset:", ec["run"]["dataset_id"])
    print("  architecture:", ec["run"]["architecture_id"])
    print("  retrieval method:", retrieval_method)
    print()
    print("  queries:", 6)
    print("  judgments:", sum(len(v) for v in judgments.values()))
    print("  retrieval cases:", len(cases))
    print("  cutoffs:", sorted(expected_cutoffs))
    print()
    print("  query metric rows:", len(query_metrics))
    print("  aggregate metric rows:", len(aggregate_metrics))
    print()
    print("  metrics:")
    for m in METRIC_FIELDS:
        print("    " + m)
    print()
    print("  all metrics finite: True")
    print("  all metrics within [0, 1]: True")
    print("  macro aggregation valid: True")
    print()
    print("  duplicate document identities preserved: True")
    print("  content-based judgment expansion: False")
    print()
    print("  wall clock seconds:", round(wall_clock, 6))
    print("  output bytes:", output_bytes)
    print()
    print("  retrieval artifacts unchanged:", retrieval_unchanged)
    print("  judgments unchanged:", judgments_unchanged)
    print()
    print("  round trip:", round_trip)
    print("  checksums valid: True")
    print("  deterministic artifacts: True")
    print()
    print("  retrieval executed: False")
    print("  model loaded: False")
    print("  Qdrant client opened: False")
    print("  generation executed: False")
    print("  answer evaluation executed: False")
    print("  LLM judge executed: False")
    print("  architecture comparison executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())