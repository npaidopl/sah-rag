"""Validate the architecture environment matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

MATRIX = Path("baseline_audit/architecture_environment_matrix.yaml")
PORTFOLIO = Path("baseline_audit/architecture_portfolio_dispositions.yaml")

EXPECTED = {
    "longrag", "raptor", "hipporag2", "lightrag", "pathrag",
    "disretrieval", "memorag", "kag",
    "metadata_rag", "hikey",
}
ALLOWED_STATUSES = {
    "EXECUTABLE_ON_CURRENT_HOST",
    "EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS",
    "REQUIRES_REMOTE_GPU",
    "REQUIRES_HOSTED_SERVICE",
    "REQUIRES_SERVICE_ENVIRONMENT",
    "REQUIRES_MULTIMODAL_CONTRACT",
    "NOT_EXECUTABLE",
}
ALLOWED_BLOCKERS = {
    "MISSING_GPU", "INSUFFICIENT_VRAM", "MISSING_SERVICE",
    "MISSING_MODEL_LICENSE", "MODEL_NOT_APPROVED", "NETWORK_POLICY_BLOCKED",
    "HOSTED_COST_NOT_APPROVED", "MULTIMODAL_CONTRACT_MISSING",
    "DATASET_SOURCE_FORMAT_UNVERIFIED", "FOREIGN_EXECUTION_BLOCKED",
    "IMPLEMENTATION_NOT_READY",
}
PILOT_ELIGIBLE_STATUSES = {
    "EXECUTABLE_ON_CURRENT_HOST",
    "EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS",
}


def main() -> int:
    for path in (MATRIX, PORTFOLIO):
        if not path.is_file():
            raise SystemExit(f"missing: {path}")

    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    portfolio = yaml.safe_load(PORTFOLIO.read_text(encoding="utf-8"))

    archs = matrix["architectures"]
    if set(archs) != EXPECTED:
        raise SystemExit("architecture set changed")

    portfolio_status = {
        aid: entry["operational_status"]
        for aid, entry in portfolio["architectures"].items()
    }

    counts = {s: 0 for s in ALLOWED_STATUSES}
    pilot_eligible_count = 0
    for aid, entry in archs.items():
        if entry["portfolio_status"] != portfolio_status[aid]:
            raise SystemExit(f"{aid}: portfolio_status mismatch with Step 75")
        status = entry["environment_status"]
        if status not in ALLOWED_STATUSES:
            raise SystemExit(f"{aid}: invalid environment_status {status}")
        counts[status] += 1
        for b in entry["blockers"]:
            if b not in ALLOWED_BLOCKERS:
                raise SystemExit(f"{aid}: invalid blocker {b}")
        if not entry.get("reason"):
            raise SystemExit(f"{aid}: reason required")
        if not entry.get("next_action"):
            raise SystemExit(f"{aid}: next_action required")
        if entry["pilot_eligible"]:
            if portfolio_status[aid] != "READY_FOR_DATA":
                raise SystemExit(f"{aid}: pilot_eligible without READY_FOR_DATA")
            if status not in PILOT_ELIGIBLE_STATUSES:
                raise SystemExit(f"{aid}: pilot_eligible with non-executable status")
            if entry["blockers"]:
                raise SystemExit(f"{aid}: pilot_eligible with blockers")
            pilot_eligible_count += 1

    if archs["pathrag"]["pilot_eligible"] is not True:
        raise SystemExit("pathrag must be pilot eligible")
    if archs["hipporag2"]["environment_status"] != "REQUIRES_REMOTE_GPU":
        raise SystemExit("hipporag2 must require remote GPU on this host")
    if archs["disretrieval"]["environment_status"] != "NOT_EXECUTABLE":
        raise SystemExit("disretrieval must be NOT_EXECUTABLE")
    if archs["kag"]["environment_status"] != "REQUIRES_SERVICE_ENVIRONMENT":
        raise SystemExit("kag must require a service environment")
    if archs["hikey"]["environment_status"] != "REQUIRES_MULTIMODAL_CONTRACT":
        raise SystemExit("hikey must require a multimodal contract")
    if "MULTIMODAL_CONTRACT_MISSING" not in archs["hikey"]["blockers"]:
        raise SystemExit("hikey must declare MULTIMODAL_CONTRACT_MISSING")

    summary = matrix["summary"]
    if summary["architecture_count"] != 10:
        raise SystemExit("summary.architecture_count must be 10")
    if summary["infrastructure_pilot_eligible"] != pilot_eligible_count:
        raise SystemExit("summary.infrastructure_pilot_eligible mismatch")
    if summary["publishable_execution_allowed"] is not False:
        raise SystemExit("summary.publishable_execution_allowed must be false")

    print("[OK] Architecture environment matrix")
    print(f"  architectures: {len(archs)}")
    print(f"  executable on current host: {counts['EXECUTABLE_ON_CURRENT_HOST']}")
    print(f"  executable with limits: {counts['EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS']}")
    print(f"  remote GPU required: {counts['REQUIRES_REMOTE_GPU']}")
    print(f"  hosted service required: {counts['REQUIRES_HOSTED_SERVICE']}")
    print(f"  service environment required: {counts['REQUIRES_SERVICE_ENVIRONMENT']}")
    print(f"  multimodal contract required: {counts['REQUIRES_MULTIMODAL_CONTRACT']}")
    print(f"  not executable: {counts['NOT_EXECUTABLE']}")
    print(f"  infrastructure-pilot eligible: {pilot_eligible_count}")
    print("  publishable execution allowed: False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
