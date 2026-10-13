"""Validate the Step 89 retrieval evaluation execution manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_evaluation_execution_manifest.yaml")
EXPECTED_ID = "pilot-bounded-retrieval-evaluation-v1"
ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_evaluation_execution_manifest",
        "artifact_type changed",
    )

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_ID, "execution_id changed")
    status = execution["status"]
    _require(status in ALLOWED_STATUSES, "status must be valid")

    approval = data["approval"]
    for flag in (
        "judgment_loading_approved",
        "retrieval_case_construction_approved",
        "evaluation_execution_approved",
        "metric_computation_approved",
        "aggregate_metric_computation_approved",
        "persistence_approved",
    ):
        _require(approval[flag] is True, "approval." + flag + " must be true")
    for flag in (
        "retrieval_execution_approved",
        "generation_execution_approved",
        "answer_evaluation_approved",
        "llm_as_judge_approved",
        "architecture_comparison_approved",
        "publishable_execution_approved",
    ):
        _require(approval[flag] is False, "approval." + flag + " must be false")

    plan = data["plan"]
    _require(plan["query_count"] == 6, "query_count must be 6")
    _require(plan["judgment_count"] == 12, "judgment_count must be 12")
    _require(plan["metric_count"] == 5, "metric_count must be 5")
    _require(list(plan["cutoffs"]) == [1, 5, 10, 100], "cutoffs must be [1, 5, 10, 100]")

    results = data["results"]
    if status == "PLANNED":
        for flag in (
            "judgments_loaded",
            "retrieval_cases_created",
            "evaluation_executed",
            "metrics_computed",
            "aggregate_metrics_computed",
            "retrieval_executed",
            "generation_executed",
            "answer_evaluation_executed",
            "llm_as_judge_executed",
            "architecture_comparison_executed",
        ):
            _require(results[flag] is False, "results." + flag + " must be false")
        for key in (
            "retrieval_case_count",
            "query_metric_count",
            "aggregate_metric_count",
            "wall_clock_seconds",
            "output_bytes",
        ):
            _require(results[key] is None, "results." + key + " must be null")
    elif status == "SUCCEEDED":
        _require(results["judgments_loaded"] is True, "judgments_loaded must be true")
        _require(results["evaluation_executed"] is True, "evaluation_executed must be true")
        _require(results["metrics_computed"] is True, "metrics_computed must be true")
        _require(
            results["aggregate_metrics_computed"] is True,
            "aggregate_metrics_computed must be true",
        )
        _require(results["retrieval_case_count"] == 6, "retrieval_case_count must be 6")
        _require(results["query_metric_count"] == 24, "query_metric_count must be 24")
        _require(results["aggregate_metric_count"] == 4, "aggregate_metric_count must be 4")
        _require(results["round_trip_valid"] is True, "round_trip_valid must be true")
        _require(results["checksums_valid"] is True, "checksums_valid must be true")
        _require(results["retrieval_executed"] is False, "retrieval_executed must be false")
        _require(results["generation_executed"] is False, "generation_executed must be false")
        _require(
            results["answer_evaluation_executed"] is False,
            "answer_evaluation_executed must be false",
        )
        _require(results["llm_as_judge_executed"] is False, "llm_as_judge_executed must be false")
        _require(
            results["architecture_comparison_executed"] is False,
            "architecture_comparison_executed must be false",
        )

    next_gate = data["next_gate"]
    _require(
        next_gate["required_step"] == "infrastructure_pilot_closure",
        "next_gate.required_step changed",
    )
    _require(next_gate["approved"] is False, "next_gate.approved must be false")

    print("[OK] Pilot evaluation execution manifest")
    print("  execution id:", execution["execution_id"])
    print("  status:", status)
    print("  evaluation approved:", approval["evaluation_execution_approved"])
    print("  metric computation approved:", approval["metric_computation_approved"])
    print("  retrieval approved:", approval["retrieval_execution_approved"])
    print("  generation approved:", approval["generation_execution_approved"])
    print("  publishable execution approved:", approval["publishable_execution_approved"])
    if status == "SUCCEEDED":
        print("  judgments loaded:", results["judgments_loaded"])
        print("  evaluation executed:", results["evaluation_executed"])
        print("  metrics computed:", results["metrics_computed"])
        print("  retrieval cases:", results["retrieval_case_count"])
        print("  query metric rows:", results["query_metric_count"])
        print("  aggregate metric rows:", results["aggregate_metric_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())