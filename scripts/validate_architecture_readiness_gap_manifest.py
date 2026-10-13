"""Validate the Step 94 readiness gap manifest."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/architecture_readiness_gap_manifest.yaml")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "architecture_readiness_gap_manifest",
             "artifact_type changed")
    e = d["execution"]
    _require(e["status"] in ("PLANNED","COMPLETE","INTERRUPTED"),
             "status must be valid")

    ap = d["approval"]
    _require(ap["static_inspection_approved"] is True,
             "static_inspection_approved must be true")
    _require(ap["readiness_planning_approved"] is True,
             "readiness_planning_approved must be true")
    for f in (
        "architecture_execution_approved","model_loading_approved",
        "llm_execution_approved","storage_creation_approved",
        "retrieval_execution_approved","evaluation_execution_approved",
    ):
        _require(ap[f] is False, "approval." + f + " must be false")

    r = d["results"]
    if e["status"] == "COMPLETE":
        _require(r["architecture_records_created"] is True,
                 "architecture_records_created must be true")
        _require(r["architecture_count"] == 11, "architecture_count must be 11")
        _require(r["implementation_sequence_created"] is True,
                 "implementation_sequence_created must be true")
        for k in ("dataset_ready_count","synthetic_validated_count",
                  "partial_count","not_started_count","deferred_count"):
            _require(isinstance(r[k], int), f"{k} must be int")

    ng = d["next_gate"]
    _require(
        ng["required_step"] == "first_architecture_readiness_completion",
        "next_gate.required_step changed",
    )
    _require(ng["approved"] is False, "next_gate.approved must be false")

    print("[OK] Architecture readiness gap manifest")
    print("  status:", e["status"])
    print("  architecture records:", r["architecture_records_created"])
    print("  architecture count:", r["architecture_count"])
    print("  implementation sequence present:",
          r["implementation_sequence_created"])
    print("  first target:", ng["target_architecture"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())