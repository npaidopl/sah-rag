"""Validate baseline_audit/longrag_storage_equivalence_report.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/longrag_storage_equivalence_report.yaml")

ALLOWED_STATUS = {"PLANNED", "PASSED", "FAILED"}
ALLOWED_CLASSIFICATION = {
    "UNRESOLVED",
    "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK",
    "STORAGE_SUBSTITUTION_BLOCKED",
}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if not PATH.exists():
        fail(f"missing: {PATH}")

    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))

    if d["artifact_type"] != "longrag_storage_equivalence_report":
        fail("wrong artifact_type")

    cmp_ = d["comparison"]
    status = cmp_["status"]
    if status not in ALLOWED_STATUS:
        fail(f"comparison.status must be in {ALLOWED_STATUS}")

    if d["reference_backend"] != "EXACT_IN_MEMORY_DENSE":
        fail("reference_backend must be EXACT_IN_MEMORY_DENSE")
    if d["candidate_backend"] != "QDRANT_LOCAL_PERSISTENCE":
        fail("candidate_backend must be QDRANT_LOCAL_PERSISTENCE")
    if d["query_count"] != 6:
        fail("query_count must be 6")

    concl = d["conclusion"]
    if concl["classification"] not in ALLOWED_CLASSIFICATION:
        fail(f"classification must be in {ALLOWED_CLASSIFICATION}")

    if status == "PASSED":
        if concl["classification"] != "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK":
            fail("PASSED requires classification=SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK")
        if concl["substitution_allowed"] is not True:
            fail("PASSED requires substitution_allowed=true")
        res = d["results"]
        for k in ("top_1_overlap_mean", "top_5_overlap_mean",
                  "top_10_overlap_mean"):
            if res[k] is None or float(res[k]) < 1.0:
                fail(f"PASSED requires {k} >= 1.0")
        if res["score_direction_match"] is not True:
            fail("PASSED requires score_direction_match=true")
        if res["canonical_lineage_match"] is not True:
            fail("PASSED requires canonical_lineage_match=true")
    elif status == "PLANNED":
        if concl["substitution_allowed"] is not False:
            fail("PLANNED requires substitution_allowed=false")

    print(f"[OK] {PATH}")
    print(f"[OK] report status={status}, classification={concl['classification']}")


if __name__ == "__main__":
    main()