"""Validate the Step 88 evaluation planning manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_evaluation_manifest.yaml")
EXPECTED_ID = "pilot-retrieval-evaluation-v1"
ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_retrieval_evaluation_manifest",
        "artifact_type changed",
    )

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_ID, "execution_id changed")
    status = execution["status"]
    _require(status in ALLOWED_STATUSES, "status must be valid")

    approval = data["approval"]
    _require(
        approval["environment_inspection_approved"] is True,
        "environment_inspection_approved must be true",
    )
    _require(
        approval["evaluation_planning_approved"] is True,
        "evaluation_planning_approved must be true",
    )
    for flag in (
        "judgment_loading_approved",
        "evaluation_execution_approved",
        "metric_computation_approved",
        "generation_execution_approved",
        "publishable_execution_approved",
    ):
        _require(approval[flag] is False, "approval." + flag + " must be false")

    plan = data["plan"]
    _require(plan["query_count"] == 6, "query_count must be 6")
    _require(plan["judgment_count"] == 12, "judgment_count must be 12")
    _require(plan["retrieval_method"] == "hybrid_rrf", "retrieval_method must be hybrid_rrf")
    _require(list(plan["cutoffs"]) == [1, 5, 10, 100], "cutoffs must be [1, 5, 10, 100]")

    results = data["results"]
    if status == "PLANNED":
        _require(results["judgments_loaded"] is False, "judgments_loaded must be false")
        _require(results["evaluation_executed"] is False, "evaluation_executed must be false")
        _require(
            results["metric_computation_executed"] is False,
            "metric_computation_executed must be false",
        )
    elif status == "SUCCEEDED":
        _require(results["judgments_loaded"] is True, "judgments_loaded must be true")
        _require(results["evaluation_executed"] is True, "evaluation_executed must be true")
        _require(results["retrieval_case_count"] == 6, "retrieval_case_count must be 6")

    next_gate = data["next_gate"]
    _require(
        next_gate["required_step"] == "bounded_retrieval_evaluation_execution",
        "next_gate.required_step changed",
    )
    _require(next_gate["approved"] is False, "next_gate.approved must be false")

    print("[OK] Pilot retrieval evaluation manifest")
    print("  status:", status)
    print("  metric inventory present:", bool(plan["metric_names"]))
    print("  judgment plan hash present:", plan["judgment_plan_hash"] is not None)
    print("  evaluation plan hash present:", plan["evaluation_plan_hash"] is not None)
    print("  evaluation approved:", approval["evaluation_execution_approved"])
    print("  generation approved:", approval["generation_execution_approved"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())