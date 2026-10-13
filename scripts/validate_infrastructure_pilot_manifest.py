"""Validate the infrastructure pilot manifest."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

MANIFEST = Path("baseline_audit/infrastructure_pilot_manifest.yaml")
HOST = Path("baseline_audit/execution_host_profile.yaml")
MATRIX = Path("baseline_audit/architecture_environment_matrix.yaml")

EXPECTED = {"longrag", "raptor", "lightrag", "pathrag"}


def main() -> int:
    for p in (MANIFEST, HOST, MATRIX):
        if not p.is_file():
            raise SystemExit(f"missing: {p}")

    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    host = yaml.safe_load(HOST.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))

    pilot = data["pilot"]
    if pilot["pilot_id"] != "infrastructure-pilot-wave1-cpu-v1":
        raise SystemExit("pilot_id changed")
    if pilot["contract_version"] != "0.1.0":
        raise SystemExit("contract_version must be 0.1.0")
    if pilot["status"] not in ("PLANNED", "CLOSED"):
        raise SystemExit("status must be PLANNED or CLOSED")
    if pilot["execution_role"] != "INFRASTRUCTURE_VALIDATION_ONLY":
        raise SystemExit("execution_role changed")
    if pilot["publishable_result"] is not False:
        raise SystemExit("publishable_result must be false")

    ds = data["dataset"]
    # After Step 78 the dataset may be materialized. The two flags
    # must agree with each other, and materialization must not
    # imply execution approval.
    if ds["dataset_materialized"] is not ds["canonical_manifest_available"]:
        raise SystemExit(
            "dataset_materialized and canonical_manifest_available "
            "must agree"
        )

    if data["host"]["profile_id"] != host["profile"]["profile_id"]:
        raise SystemExit("host profile_id mismatch")

    archs = data["architectures"]
    if set(archs) != EXPECTED:
        raise SystemExit("architecture set changed")
    for aid, entry in archs.items():
        if entry["included"] is not True:
            raise SystemExit(f"{aid}: included must be true")
        if entry["pilot_eligible"] is not True:
            raise SystemExit(f"{aid}: pilot_eligible must be true")
        if entry["experiment_cell_id"] is not None:
            raise SystemExit(f"{aid}: experiment_cell_id must be null")
        if entry["plan_id"] is not None:
            raise SystemExit(f"{aid}: plan_id must be null")
        if entry["status"] != "NOT_PLANNED":
            raise SystemExit(f"{aid}: status must be NOT_PLANNED")
        # environment status must match matrix
        if entry["environment_status"] != matrix["architectures"][aid]["environment_status"]:
            raise SystemExit(f"{aid}: environment_status mismatch")

    counts = data["planned_counts"]
    if counts["architecture_count"] != 4:
        raise SystemExit("architecture_count must be 4")
    if counts["experiment_cell_count"] != 4:
        raise SystemExit("experiment_cell_count must be 4")
    if counts["stage_count_per_architecture"] != 8:
        raise SystemExit("stage_count_per_architecture must be 8")
    if counts["total_stage_count"] != 32:
        raise SystemExit("total_stage_count must be 32")

    st = data["execution_state"]
    for k in (
        "architecture_execution_performed", "model_execution_performed",
        "dataset_loaded", "evaluation_execution_performed",
        "hosted_requests_performed", "final_generation_performed",
    ):
        if st[k] is not False:
            raise SystemExit(f"execution_state.{k} must be false")

    ap = data["approval"]
    # materialization_approved may be true after Step 78.
    # execution_approved becomes true only after the pilot closes
    # (Step 90). publishable_execution_approved must always be
    # false. future_execution_approved must always be false.
    if pilot["status"] == "PLANNED":
        if ap["execution_approved"] is not False:
            raise SystemExit("execution_approved must be false while PLANNED")
    else:
        if ap["execution_approved"] is not True:
            raise SystemExit("execution_approved must be true after CLOSED")
    if ap["publishable_execution_approved"] is not False:
        raise SystemExit("publishable_execution_approved must be false")
    if "future_execution_approved" in ap and ap["future_execution_approved"] is not False:
        raise SystemExit("future_execution_approved must be false")
    if (
        "real_dataset_execution_approved" in ap
        and ap["real_dataset_execution_approved"] is not False
    ):
        raise SystemExit("real_dataset_execution_approved must be false")

    print("[OK] Infrastructure pilot manifest")
    print(f"  pilot id: {pilot['pilot_id']}")
    print(f"  status: {pilot['status']}")
    print(f"  role: {pilot['execution_role']}")
    print(f"  architectures: {len(archs)}")
    print(f"  experiment cells: {counts['experiment_cell_count']}")
    print(f"  total stages: {counts['total_stage_count']}")
    print(f"  dataset materialized: {ds['dataset_materialized']}")
    print(f"  execution approved: {ap['execution_approved']}")
    print(f"  publishable execution approved: {ap['publishable_execution_approved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
