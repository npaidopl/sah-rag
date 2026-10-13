"""Validate the Step 93 fair benchmark manifest."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/fair_benchmark_manifest.yaml")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "fair_benchmark_manifest", "artifact_type changed")
    e = d["execution"]
    _require(e["status"] in ("PLANNED","SUCCEEDED","INTERRUPTED"), "status must be valid")

    ap = d["approval"]
    for f in ("methodology_planning_approved","architecture_classification_approved",
              "component_substitution_planning_approved","dataset_task_mapping_approved",
              "statistical_plan_approved"):
        _require(ap[f] is True, "approval." + f + " must be true")
    for f in ("architecture_implementation_approved","tokenizer_execution_approved",
              "model_execution_approved","embedding_execution_approved",
              "retrieval_execution_approved","evaluation_execution_approved"):
        _require(ap[f] is False, "approval." + f + " must be false")

    r = d["results"]
    if e["status"] == "SUCCEEDED":
        _require(r["architecture_records_created"] is True, "architectures must be created")
        _require(r["dataset_task_records_created"] is True, "dataset tasks must be created")
        _require(r["precision_protocol_frozen"] is True, "precision protocol must be frozen")
        _require(r["statistical_plan_frozen"] is True, "statistical plan must be frozen")
        _require(r["controlled_track_count"] == 6, "controlled_track_count must be 6")
        _require(
            r["native_track_count"] == 1,
            "native_track_count must be 1 (currently eligible)",
        )
        _require(
            r.get("native_track_potential_candidate_count") == 4,
            "native_track_potential_candidate_count must be 4",
        )
        _require(r["both_track_count"] == 1, "both_track_count must be 1")
        _require(r["deferred_count"] == 5, "deferred_count must be 5")

    ng = d["next_gate"]
    _require(ng["required_step"] == "per_architecture_readiness_gap_registry",
             "next_gate.required_step changed")
    _require(ng["approved"] is False, "next_gate.approved must be false")

    print("[OK] Fair benchmark manifest")
    print("  status:", e["status"])
    print("  controlled_track_count:", r["controlled_track_count"])
    print("  native_track_count:", r["native_track_count"])
    print("  both_track_count:", r["both_track_count"])
    print("  deferred_count:", r["deferred_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())