"""Validate the Step 94 readiness gap registry."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/architecture_readiness_gap_registry.yaml")

ARCHS = ("sah_rag","longrag","raptor","hipporag2","lightrag","pathrag",
         "disretrieval","memorag","kag","metadata_rag","hikey")

REQUIRED = (
    "architecture_id","implementation_status","fidelity_class",
    "controlled_track_status","native_track_status","readiness_gate",
    "current_artifacts","verified_components","missing_components",
    "architecture_defining_components","controlled_substitutions",
    "llm_dependencies","storage_policy","canonical_input_status",
    "retrieval_output_status","synthetic_acceptance_status",
    "dataset_readiness_status","blockers","next_action_class",
    "next_action","recommended_order",
)


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "architecture_readiness_gap_registry",
             "artifact_type changed")
    r = d["registry"]
    _require(r["architecture_count"] == 11, "architecture_count must be 11")
    _require(r["status"] == "COMPLETE", "status must be COMPLETE")

    a = d["architectures"]
    _require(set(a.keys()) == set(ARCHS), "architecture set changed")

    counts = {
        "DATASET_READY": 0, "SYNTHETIC_VALIDATED": 0, "COMPONENT_TESTED": 0,
        "PARTIAL": 0, "STATIC_SPEC_COMPLETE": 0, "NOT_STARTED": 0,
    }
    deferred = 0

    for name in ARCHS:
        rec = a[name]
        for f in REQUIRED:
            _require(f in rec, f"{name}: missing field {f}")
        _require(rec["blockers"] is not None, f"{name}: blockers must exist")
        _require(
            isinstance(rec["next_action"], str) and rec["next_action"].strip(),
            f"{name}: next_action must be non-empty",
        )
        _require(
            isinstance(rec["recommended_order"], int),
            f"{name}: recommended_order must be int",
        )
        st = rec["implementation_status"]
        if st in counts:
            counts[st] += 1
        if rec["readiness_gate"] == "DEFERRED_WITH_REASON":
            deferred += 1

    s = d["summary"]
    _require(s["dataset_ready_count"] == counts["DATASET_READY"],
             "dataset_ready_count mismatch")
    _require(s["synthetic_validated_count"] == counts["SYNTHETIC_VALIDATED"],
             "synthetic_validated_count mismatch")
    _require(s["partial_count"] == counts["PARTIAL"], "partial_count mismatch")
    _require(s["not_started_count"] == counts["NOT_STARTED"],
             "not_started_count mismatch")
    _require(
        s["native_track_current_eligible_count"] == 1,
        "native_track_current_eligible_count must be 1",
    )
    _require(
        s["native_track_potential_candidate_count"] == 4,
        "native_track_potential_candidate_count must be 4",
    )

    print("[OK] Architecture readiness gap registry")
    print("  architectures:", len(a))
    print("  dataset_ready:", counts["DATASET_READY"])
    print("  synthetic_validated:", counts["SYNTHETIC_VALIDATED"])
    print("  partial:", counts["PARTIAL"])
    print("  not_started:", counts["NOT_STARTED"])
    print("  deferred:", deferred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())