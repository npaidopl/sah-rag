"""Validate baseline_audit/longrag_readiness_declaration.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/longrag_readiness_declaration.yaml")

PENDING_STATUS = "PENDING_SYNTHETIC_ACCEPTANCE"
PASSED_STATUS = "SYNTHETIC_ACCEPTANCE_PASSED"

VALIDATED_FLAGS = (
    "canonical_input_validated",
    "long_unit_construction_validated",
    "controlled_embedding_validated",
    "persistent_state_validated",
    "state_reopen_validated",
    "retrieval_validated",
    "storage_equivalence_validated",
    "canonical_identity_expansion_validated",
    "evaluation_case_mapping_validated",
)


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if not PATH.exists():
        fail(f"missing: {PATH}")

    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))

    if d["artifact_type"] != "architecture_readiness_declaration":
        fail("wrong artifact_type")

    arch = d["architecture"]
    if arch["architecture_id"] != "longrag":
        fail("architecture_id must be longrag")
    if arch["fidelity_class"] != "CONTROLLED_ADAPTATION":
        fail("fidelity_class must be CONTROLLED_ADAPTATION")

    rd = d["readiness"]
    status = rd["status"]
    if status not in (PENDING_STATUS, PASSED_STATUS):
        fail(f"status must be {PENDING_STATUS} or {PASSED_STATUS}")

    if rd["generation_disabled"] is not True:
        fail("generation_disabled must be true")

    if status == PENDING_STATUS:
        for k in VALIDATED_FLAGS:
            if rd[k] is not False:
                fail(f"PENDING: {k} must be false")
        if rd["dataset_ready"] is not False:
            fail("PENDING: dataset_ready must be false")
    else:
        for k in VALIDATED_FLAGS:
            if rd[k] is not True:
                fail(f"PASSED: {k} must be true")
        if rd["dataset_ready"] is not True:
            fail("PASSED: dataset_ready must be true")

    claims = d["claims"]
    if claims["faithful_reproduction"] is not False:
        fail("faithful_reproduction must be false")
    if claims["controlled_adaptation"] is not True:
        fail("controlled_adaptation must be true")
    if claims["native_embedder_equivalence"] is not False:
        fail("native_embedder_equivalence must be false")
    if claims["publishable_result"] is not False:
        fail("publishable_result must be false")

    tr = d["track"]
    if tr["controlled_track_eligible"] is not True:
        fail("controlled_track_eligible must be true")
    if tr["native_track_eligible"] is not False:
        fail("native_track_eligible must be false")

    print(f"[OK] {PATH}")
    print(f"[OK] readiness status={status}, dataset_ready={rd['dataset_ready']}")


if __name__ == "__main__":
    main()