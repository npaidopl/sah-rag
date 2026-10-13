"""Validate the execution environment contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT = Path("baseline_audit/execution_environment_contract.yaml")

EXPECTED_STATUSES = {
    "EXECUTABLE_ON_CURRENT_HOST",
    "EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS",
    "REQUIRES_REMOTE_GPU",
    "REQUIRES_HOSTED_SERVICE",
    "REQUIRES_SERVICE_ENVIRONMENT",
    "REQUIRES_MULTIMODAL_CONTRACT",
    "NOT_EXECUTABLE",
}

EXPECTED_BLOCKERS = {
    "MISSING_GPU",
    "INSUFFICIENT_VRAM",
    "MISSING_SERVICE",
    "MISSING_MODEL_LICENSE",
    "MODEL_NOT_APPROVED",
    "NETWORK_POLICY_BLOCKED",
    "HOSTED_COST_NOT_APPROVED",
    "MULTIMODAL_CONTRACT_MISSING",
    "DATASET_SOURCE_FORMAT_UNVERIFIED",
    "FOREIGN_EXECUTION_BLOCKED",
    "IMPLEMENTATION_NOT_READY",
}


def main() -> int:
    if not CONTRACT.is_file():
        raise SystemExit(f"missing contract: {CONTRACT}")
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    c = data["contract"]
    if c["id"] != "architecture_execution_environment":
        raise SystemExit("contract.id changed")
    if c["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if c["status"] != "FROZEN_FOR_IMPLEMENTATION":
        raise SystemExit("contract.status must be FROZEN_FOR_IMPLEMENTATION")

    scope = data["scope"]
    if scope["architecture_count"] != 10:
        raise SystemExit("scope.architecture_count must be 10")

    statuses = set(data["execution_status_vocabulary"]["allowed"])
    if statuses != EXPECTED_STATUSES:
        raise SystemExit("status vocabulary changed")

    blockers = set(data["blocker_vocabulary"]["allowed"])
    if blockers != EXPECTED_BLOCKERS:
        raise SystemExit("blocker vocabulary changed")

    policy = data["pilot_policy"]
    if policy["publishable_execution_allowed"] is not False:
        raise SystemExit("publishable_execution_allowed must be false")
    if policy["infrastructure_only"] is not True:
        raise SystemExit("infrastructure_only must be true")
    if set(policy["eligible_portfolio_status"]) != {"READY_FOR_DATA"}:
        raise SystemExit("eligible_portfolio_status must be READY_FOR_DATA only")

    safety = data["safety"]
    for k in (
        "foreign_blocked_repositories_must_not_execute",
        "unapproved_hosted_services_must_not_execute",
        "missing_model_license_blocks_execution",
        "missing_multimodal_contract_blocks_hikey",
        "no_secrets_written_to_artifacts",
    ):
        if safety[k] is not True:
            raise SystemExit(f"safety.{k} must be true")

    print("[OK] Execution environment contract")
    print(f"  id: {c['id']}")
    print(f"  version: {c['version']}")
    print(f"  architecture count: {scope['architecture_count']}")
    print(f"  environment statuses: {len(statuses)}")
    print(f"  blocker types: {len(blockers)}")
    print(f"  publishable execution: {policy['publishable_execution_allowed']}")
    print(f"  infrastructure-only pilot: {policy['infrastructure_only']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
