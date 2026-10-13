"""Validate the Step 90 infrastructure pilot closure contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/infrastructure_pilot_closure_contract.yaml")

EXPECTED_CHAIN_LENGTH = 13
EXPECTED_STEPS = (77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "contract must be a mapping")
    _require(
        data.get("artifact_type") == "infrastructure_pilot_closure_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_evidence_freeze",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")
    _require(
        contract["status"] == "APPROVED_FOR_CLOSURE",
        "contract.status must be APPROVED_FOR_CLOSURE",
    )

    scope = data["scope"]
    _require(scope["synthetic_data_only"] is True, "synthetic_data_only must be true")
    _require(scope["real_dataset_used"] is False, "real_dataset_used must be false")
    _require(
        scope["publishable_execution"] is False,
        "publishable_execution must be false",
    )
    _require(scope["first_step"] == 77, "first_step must be 77")
    _require(scope["last_execution_step"] == 89, "last_execution_step must be 89")
    _require(scope["closure_step"] == 90, "closure_step must be 90")

    chain = data["commit_chain"]
    for step in EXPECTED_STEPS:
        key = f"step_{step}"
        _require(key in chain, "commit_chain missing " + key)
        sha = chain[key]
        _require(isinstance(sha, str) and len(sha) == 40, key + " must be 40-char SHA")

    statuses = data["required_stage_statuses"]
    for name, value in statuses.items():
        _require(
            value in ("COMPLETE", "SUCCEEDED"),
            "required_stage_statuses." + name + " must be COMPLETE or SUCCEEDED",
        )

    exec_chain = data["required_execution_chain"]
    _require(
        len(exec_chain) == EXPECTED_CHAIN_LENGTH,
        "required_execution_chain must have " + str(EXPECTED_CHAIN_LENGTH) + " entries",
    )

    closure = data["closure"]
    _require(
        closure["repeat_execution_required"] is False,
        "repeat_execution_required must be false",
    )
    _require(
        closure["runtime_artifact_validation_required"] is True,
        "runtime_artifact_validation_required must be true",
    )
    _require(
        closure["checksum_validation_required"] is True,
        "checksum_validation_required must be true",
    )
    _require(
        closure["commit_chain_validation_required"] is True,
        "commit_chain_validation_required must be true",
    )
    _require(
        closure["source_immutability_validation_required"] is True,
        "source_immutability_validation_required must be true",
    )

    safety = data["safety"]
    for flag in (
        "generation_executed",
        "answer_evaluation_executed",
        "llm_as_judge_executed",
        "architecture_comparison_executed",
        "hosted_requests_executed",
        "network_access_during_bounded_execution",
        "foreign_code_execution_for_pilot",
        "real_dataset_execution",
        "publishable_metrics_produced",
        "effectiveness_claims_allowed",
    ):
        _require(safety[flag] is False, "safety." + flag + " must be false")

    gates = data["quality_gates"]
    _require(
        gates["full_test_suite_minimum"] >= 878,
        "full_test_suite_minimum must be >= 878",
    )
    for flag in (
        "ruff_required",
        "mypy_src_required",
        "pip_check_required",
        "git_diff_check_required",
    ):
        _require(gates[flag] is True, "quality_gates." + flag + " must be true")

    print("[OK] Infrastructure pilot closure contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  pilot:", data["pilot"]["pilot_id"])
    print("  dataset:", data["pilot"]["dataset_id"])
    print("  synthetic only:", scope["synthetic_data_only"])
    print("  repeat execution required:", closure["repeat_execution_required"])
    print("  publishable execution:", scope["publishable_execution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())