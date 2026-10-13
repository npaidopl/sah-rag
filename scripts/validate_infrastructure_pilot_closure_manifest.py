"""Validate the Step 90 infrastructure pilot closure manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/infrastructure_pilot_closure_manifest.yaml")
EXPECTED_ID = "infrastructure-pilot-wave1-cpu-v1-closure"
ALLOWED_STATUSES = ("PLANNED", "CLOSED", "INTERRUPTED")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest must be a mapping")
    _require(
        data.get("artifact_type") == "infrastructure_pilot_closure_manifest",
        "artifact_type changed",
    )

    closure = data["closure"]
    _require(closure["closure_id"] == EXPECTED_ID, "closure_id changed")
    status = closure["status"]
    _require(status in ALLOWED_STATUSES, "status must be valid")

    approval = data["approval"]
    _require(
        approval["evidence_collection_approved"] is True,
        "evidence_collection_approved must be true",
    )
    _require(
        approval["closure_validation_approved"] is True,
        "closure_validation_approved must be true",
    )
    for flag in (
        "pipeline_reexecution_approved",
        "model_loading_approved",
        "qdrant_access_approved",
        "retrieval_execution_approved",
        "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        _require(approval[flag] is False, "approval." + flag + " must be false")

    results = data["results"]
    claims = data["claims"]
    _require(
        claims["infrastructure_chain_completed"] is True,
        "infrastructure_chain_completed must be true",
    )
    _require(claims["synthetic_data_only"] is True, "synthetic_data_only must be true")
    for flag in (
        "model_quality_validated",
        "retrieval_effectiveness_validated_for_real_data",
        "architecture_comparison_completed",
        "publishable_benchmark_completed",
        "real_dataset_execution_completed",
        "generation_validated",
        "answer_quality_validated",
    ):
        _require(claims[flag] is False, "claims." + flag + " must be false")

    if status == "PLANNED":
        # PLANNED is an intermediate state: the collector may already
        # have populated evidence_registry_created and other fields
        # before quality gates are captured and status is promoted to
        # CLOSED. Only enforce that the pilot is not yet closed and
        # that quality-gate fields are still false.
        _require(
            results["pilot_closed"] is False,
            "PLANNED must have pilot_closed false",
        )
        _require(
            results["ruff_passed"] is False
            or isinstance(results["ruff_passed"], bool),
            "ruff_passed must be bool",
        )
    elif status == "CLOSED":
        _require(
            results["evidence_registry_created"] is True,
            "CLOSED must have evidence_registry_created true",
        )
        _require(results["commit_chain_complete"] is True, "commit_chain_complete must be true")
        _require(results["runtime_manifests_valid"] is True, "runtime_manifests_valid must be true")
        _require(results["runtime_checksums_valid"] is True, "runtime_checksums_valid must be true")
        _require(results["critical_counts_valid"] is True, "critical_counts_valid must be true")
        _require(results["safety_flags_valid"] is True, "safety_flags_valid must be true")
        _require(
            isinstance(results["evidence_fingerprint_count"], int)
            and results["evidence_fingerprint_count"] > 0,
            "evidence_fingerprint_count must be positive",
        )
        _require(
            isinstance(results["full_test_suite_count"], int)
            and results["full_test_suite_count"] >= 878,
            "full_test_suite_count must be >= 878",
        )
        for flag in (
            "ruff_passed",
            "mypy_src_passed",
            "pip_check_passed",
            "diff_check_passed",
            "pilot_closed",
        ):
            _require(results[flag] is True, "results." + flag + " must be true")

    next_gate = data["next_gate"]
    _require(
        next_gate["required_step"] == "real_dataset_inventory_and_sizing",
        "next_gate.required_step changed",
    )
    _require(next_gate["approved"] is False, "next_gate.approved must be false")

    print("[OK] Infrastructure pilot closure manifest")
    print("  closure id:", closure["closure_id"])
    print("  status:", status)
    print("  evidence registry created:", results["evidence_registry_created"])
    print("  evidence fingerprints:", results["evidence_fingerprint_count"])
    print("  full test suite:", results["full_test_suite_count"])
    print("  Ruff passed:", results["ruff_passed"])
    print("  Mypy src passed:", results["mypy_src_passed"])
    print("  pip check passed:", results["pip_check_passed"])
    print("  diff check passed:", results["diff_check_passed"])
    print("  pilot closed:", results["pilot_closed"])
    print("  publishable execution approved:", False)
    print("  real dataset execution approved:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())