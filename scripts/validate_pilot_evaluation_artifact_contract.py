"""Validate the Step 88 evaluation artifact contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_evaluation_artifact_contract.yaml")

REQUIRED_FILES = {
    "retrieval_cases.jsonl",
    "query_metrics.jsonl",
    "aggregate_metrics.json",
    "evaluation.manifest.json",
    "checksums.json",
}

REQUIRED_PROHIBITED = {
    "generated_answers",
    "reference_answer_text",
    "answer_correctness",
    "architecture_ranking",
    "publishable_comparison",
    "llm_judge_results",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "artifact contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_evaluation_artifact_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_retrieval_evaluation_artifacts",
        "contract.id changed",
    )

    outputs = data["outputs"]
    planned = set(outputs["planned_files"])
    _require(planned == REQUIRED_FILES, "planned_files mismatch")

    rc = data["retrieval_cases"]
    for name in ("query_id", "ranked_doc_ids", "relevant_doc_ids", "cutoffs", "metadata"):
        _require(name in rc["required_fields"], "retrieval_cases missing " + name)

    qm = data["query_metrics"]
    for name in ("query_id", "metric_name", "cutoff", "value", "metadata"):
        _require(name in qm["required_fields"], "query_metrics missing " + name)

    am = data["aggregate_metrics"]
    for name in ("metric_name", "cutoff", "query_count", "mean", "minimum", "maximum", "metadata"):
        _require(name in am["required_fields"], "aggregate_metrics missing " + name)

    ordering = data["ordering"]
    _require(ordering["retrieval_cases"] == ["query_id"], "retrieval_cases ordering changed")
    _require(
        ordering["query_metrics"] == ["query_id", "metric_name", "cutoff"],
        "query_metrics ordering changed",
    )
    _require(
        ordering["aggregate_metrics"] == ["metric_name", "cutoff"],
        "aggregate_metrics ordering changed",
    )

    prohibited = set(data["prohibited_outputs"])
    missing = REQUIRED_PROHIBITED - prohibited
    _require(not missing, "prohibited_outputs missing: " + ", ".join(sorted(missing)))

    ser = data["serialization"]
    _require(ser["encoding"] == "UTF-8", "encoding must be UTF-8")
    _require(ser["line_endings"] == "LF", "line_endings must be LF")
    _require(ser["checksum_algorithm"] == "SHA-256", "checksum_algorithm must be SHA-256")
    _require(ser["atomic_writes_required"] is True, "atomic_writes_required must be true")

    print("[OK] Pilot evaluation artifact contract")
    print("  retrieval cases: planned")
    print("  query metrics: planned")
    print("  aggregate metrics: planned")
    print("  generated answers prohibited: True")
    print("  architecture ranking prohibited: True")
    print("  publishable comparison prohibited: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())