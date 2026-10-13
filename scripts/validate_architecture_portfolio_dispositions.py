"""Validate the architecture portfolio terminal disposition registry."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

PORTFOLIO_PATH = Path("baseline_audit/architecture_portfolio_dispositions.yaml")

EXPECTED_ARCHITECTURES = {
    "longrag", "raptor", "hipporag2", "lightrag", "pathrag",
    "disretrieval", "memorag", "kag",
    "metadata_rag", "hikey",
}
EXPECTED_WAVE1 = {"longrag", "raptor", "hipporag2", "lightrag", "pathrag"}
EXPECTED_WAVE2 = {"disretrieval", "memorag", "kag"}
EXPECTED_WAVE3 = {"metadata_rag", "hikey"}

EXPECTED_READY = {"longrag", "raptor", "hipporag2", "lightrag", "pathrag"}
EXPECTED_CONDITIONAL = {"disretrieval", "memorag", "metadata_rag"}
EXPECTED_DEFERRED = {"kag", "hikey"}
EXPECTED_BLOCKED: set[str] = set()

ALLOWED_STATUSES = {
    "READY_FOR_DATA",
    "CONDITIONAL_GO",
    "DEFERRED_WITH_REASON",
    "BLOCKED_WITH_REASON",
}
PROHIBITED_STATUSES = {
    "NOT_STARTED",
    "UNKNOWN",
    "PENDING",
    "PENDING_UNBOUNDED",
    "UNDECIDED",
}


def _load() -> dict:
    if not PORTFOLIO_PATH.is_file():
        raise SystemExit(f"missing portfolio: {PORTFOLIO_PATH}")
    return yaml.safe_load(PORTFOLIO_PATH.read_text(encoding="utf-8"))


def main() -> int:
    data = _load()

    portfolio = data["portfolio"]
    if portfolio["registry_architecture_count"] != 10:
        raise SystemExit("registry_architecture_count must be 10")
    if portfolio["disposition_count"] != 10:
        raise SystemExit("disposition_count must be 10")
    if portfolio["publishable_execution_allowed"] is not False:
        raise SystemExit("publishable_execution_allowed must be false")
    if portfolio["infrastructure_only_pilot_allowed"] is not True:
        raise SystemExit("infrastructure_only_pilot_allowed must be true")
    if portfolio["audit_complete"] is not True:
        raise SystemExit("audit_complete must be true")
    if portfolio["terminal_disposition_freeze_complete"] is not True:
        raise SystemExit("terminal_disposition_freeze_complete must be true")

    allowed = set(data["status_vocabulary"]["allowed"])
    prohibited = set(data["status_vocabulary"]["prohibited"])
    if allowed != ALLOWED_STATUSES:
        raise SystemExit("status_vocabulary.allowed changed")
    if prohibited != PROHIBITED_STATUSES:
        raise SystemExit("status_vocabulary.prohibited changed")

    archs = data["architectures"]
    if set(archs) != EXPECTED_ARCHITECTURES:
        raise SystemExit("architecture set changed")

    ready, conditional, deferred, blocked = set(), set(), set(), set()
    for arch_id, entry in archs.items():
        status = entry["operational_status"]
        if status in PROHIBITED_STATUSES:
            raise SystemExit(f"{arch_id}: prohibited status {status}")
        if status not in ALLOWED_STATUSES:
            raise SystemExit(f"{arch_id}: invalid status {status}")
        if not entry.get("reason"):
            raise SystemExit(f"{arch_id}: missing reason")
        if not entry.get("next_action"):
            raise SystemExit(f"{arch_id}: missing next_action")
        if entry["wave"] not in (1, 2, 3):
            raise SystemExit(f"{arch_id}: invalid wave")

        if status == "READY_FOR_DATA":
            ready.add(arch_id)
            rd = entry.get("readiness_declaration")
            if not rd:
                raise SystemExit(f"{arch_id}: READY_FOR_DATA without readiness_declaration")
            if not Path(rd).is_file():
                raise SystemExit(f"{arch_id}: readiness_declaration not on disk: {rd}")
        else:
            if entry.get("readiness_declaration") is not None:
                raise SystemExit(
                    f"{arch_id}: non-ready architecture must not have readiness_declaration"
                )
            aa = entry.get("audit_artifact")
            if not aa:
                raise SystemExit(f"{arch_id}: missing audit_artifact")
            if not Path(aa).is_file():
                raise SystemExit(f"{arch_id}: audit_artifact not on disk: {aa}")

        if status == "CONDITIONAL_GO":
            conditional.add(arch_id)
        elif status == "DEFERRED_WITH_REASON":
            deferred.add(arch_id)
        elif status == "BLOCKED_WITH_REASON":
            blocked.add(arch_id)

    if ready != EXPECTED_READY:
        raise SystemExit(f"ready set mismatch: {sorted(ready)}")
    if conditional != EXPECTED_CONDITIONAL:
        raise SystemExit(f"conditional set mismatch: {sorted(conditional)}")
    if deferred != EXPECTED_DEFERRED:
        raise SystemExit(f"deferred set mismatch: {sorted(deferred)}")
    if blocked != EXPECTED_BLOCKED:
        raise SystemExit(f"blocked set mismatch: {sorted(blocked)}")

    if (set(archs) & EXPECTED_WAVE1) != EXPECTED_WAVE1:
        raise SystemExit("wave 1 set changed")
    if (set(archs) & EXPECTED_WAVE2) != EXPECTED_WAVE2:
        raise SystemExit("wave 2 set changed")
    if (set(archs) & EXPECTED_WAVE3) != EXPECTED_WAVE3:
        raise SystemExit("wave 3 set changed")

    summary = data["summary"]
    counts = {
        "ready_for_data_count": len(ready),
        "conditional_go_count": len(conditional),
        "deferred_with_reason_count": len(deferred),
        "blocked_with_reason_count": len(blocked),
    }
    for key, value in counts.items():
        if summary[key] != value:
            raise SystemExit(f"summary.{key} mismatch: {summary[key]} != {value}")
    if summary["total_count"] != 10:
        raise SystemExit("summary.total_count must be 10")

    # Cross-cutting invariants
    if archs["pathrag"]["foreign_execution_status"] != "BLOCKED":
        raise SystemExit("pathrag foreign_execution_status must be BLOCKED")
    if archs["pathrag"]["execution_target"] != "project_owned_comparator":
        raise SystemExit("pathrag execution_target must be project_owned_comparator")

    if archs["disretrieval"]["foreign_execution_status"] != "BLOCKED":
        raise SystemExit("disretrieval foreign_execution_status must be BLOCKED")

    memorag_text = archs["memorag"]["reason"] + archs["memorag"]["next_action"]
    if "model" not in memorag_text.lower():
        raise SystemExit("memorag reason/next_action must mention model-license gate")

    kag_reason = archs["kag"]["reason"].lower()
    if "openspg" not in kag_reason and "service environment" not in kag_reason:
        raise SystemExit("kag reason must mention the service environment")

    if archs["metadata_rag"]["foreign_execution_status"] != "NOT_APPLICABLE":
        raise SystemExit("metadata_rag foreign_execution_status must be NOT_APPLICABLE")

    hikey_next = archs["hikey"]["next_action"]
    if "0.2.0" not in hikey_next:
        raise SystemExit("hikey next_action must reference contract version 0.2.0")

    gate = data["execution_gate"]
    if gate["publishable_execution_allowed"] is not False:
        raise SystemExit("execution_gate.publishable_execution_allowed must be false")
    if not gate["blockers"]:
        raise SystemExit("execution_gate.blockers must not be empty")
    if not gate["allowed_now"]:
        raise SystemExit("execution_gate.allowed_now must not be empty")
    if not gate["prohibited_now"]:
        raise SystemExit("execution_gate.prohibited_now must not be empty")

    re_eval = data["re_evaluation_policy"]
    if re_eval["semantic_status_changes_require_new_registry_version"] is not True:
        raise SystemExit(
            "re_evaluation_policy must require a new registry version"
        )

    print("[OK] Architecture portfolio terminal dispositions")
    print(f"  registry architectures: {portfolio['registry_architecture_count']}")
    print(f"  dispositions: {portfolio['disposition_count']}")
    print(f"  READY_FOR_DATA: {len(ready)}")
    print(f"  CONDITIONAL_GO: {len(conditional)}")
    print(f"  DEFERRED_WITH_REASON: {len(deferred)}")
    print(f"  BLOCKED_WITH_REASON: {len(blocked)}")
    print(f"  publishable execution allowed: {portfolio['publishable_execution_allowed']}")
    print(f"  infrastructure-only pilot allowed: {portfolio['infrastructure_only_pilot_allowed']}")
    print("  all slots have terminal dispositions: True")
    print("  prohibited statuses present: False")
    print()
    print("  inventory:")
    for arch_id in sorted(archs):
        print(f"  - {arch_id}: {archs[arch_id]['operational_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
