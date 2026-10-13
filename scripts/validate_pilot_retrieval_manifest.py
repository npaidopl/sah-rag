"""Validate the Step 86 hybrid retrieval manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_retrieval_manifest.yaml")

EXPECTED_ID = "pilot-hybrid-retrieval-v1"
ALLOWED_STATUSES = ("PLANNED", "SUCCEEDED", "INTERRUPTED")

FORBIDDEN_APPROVALS = (
    "query_embedding_approved",
    "dense_retrieval_approved",
    "sparse_retrieval_approved",
    "hybrid_retrieval_approved",
    "parent_aggregation_approved",
    "document_expansion_approved",
    "evaluation_approved",
    "generation_approved",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_hybrid_retrieval_manifest",
        "artifact_type changed",
    )

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_ID, "execution_id changed")
    _require(execution["contract_version"] == "0.1.0", "contract_version must be 0.1.0")
    status = execution["status"]
    _require(status in ALLOWED_STATUSES, "status must be one of " + ", ".join(ALLOWED_STATUSES))

    approval = data["approval"]
    _require(
        approval["environment_inspection_approved"] is True,
        "environment_inspection_approved must be true",
    )
    _require(
        approval["retrieval_planning_approved"] is True,
        "retrieval_planning_approved must be true",
    )
    for flag in FORBIDDEN_APPROVALS:
        _require(approval[flag] is False, "approval." + flag + " must be false")

    plan = data["plan"]
    _require(plan["query_count"] == 6, "query_count must be 6")
    _require(plan["collection_point_count"] == 12, "collection_point_count must be 12")
    for key in (
        "candidate_limit",
        "child_result_limit",
        "parent_result_limit",
        "document_result_limit",
    ):
        _require(plan[key] == 12, "plan." + key + " must be 12")

    results = data["results"]
    _require(results["model_loaded"] is False, "model_loaded must be false")
    _require(results["retrieval_executed"] is False, "retrieval_executed must be false")
    _require(results["evaluation_executed"] is False, "evaluation_executed must be false")
    _require(results["generation_executed"] is False, "generation_executed must be false")

    next_gate = data["next_gate"]
    _require(
        next_gate["required_step"] == "bounded_hybrid_retrieval_execution",
        "next_gate.required_step changed",
    )
    _require(next_gate["approved"] is False, "next_gate.approved must be false")

    print("[OK] Pilot hybrid retrieval manifest")
    print("  execution id:", execution["execution_id"])
    print("  status:", status)
    print("  query plan hash present:", plan["query_plan_hash"] is not None)
    print("  retrieval plan hash present:", plan["retrieval_plan_hash"] is not None)
    print("  query embedding approved:", approval["query_embedding_approved"])
    print("  hybrid retrieval approved:", approval["hybrid_retrieval_approved"])
    print("  evaluation approved:", approval["evaluation_approved"])
    print("  generation approved:", approval["generation_approved"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())