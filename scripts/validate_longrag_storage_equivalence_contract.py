"""Validate baseline_audit/longrag_storage_equivalence_contract.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/longrag_storage_equivalence_contract.yaml")

REQUIRED_TOP_KEYS = {
    "artifact_type", "artifact_version", "contract", "controlled_storage",
    "reference_storage", "identity", "fixture", "comparison",
    "acceptance", "classification", "qdrant_exactness_caveat",
}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if not PATH.exists():
        fail(f"missing: {PATH}")

    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))

    missing = REQUIRED_TOP_KEYS - set(d.keys())
    if missing:
        fail(f"missing top-level keys: {sorted(missing)}")

    if d["artifact_type"] != "longrag_storage_equivalence_contract":
        fail("wrong artifact_type")

    if d["contract"]["status"] != "APPROVED_FOR_BOUNDED_EXECUTION":
        fail("contract.status must be APPROVED_FOR_BOUNDED_EXECUTION")

    cs = d["controlled_storage"]
    if cs["backend"] != "Qdrant":
        fail("controlled_storage.backend must be Qdrant")
    if cs["vector_name"] != "dense":
        fail("vector_name must be dense")
    if cs["dimension"] != 1024:
        fail("dimension must be 1024")
    if cs["distance"] != "COSINE":
        fail("distance must be COSINE")
    for k in ("approximate_search_allowed", "quantization_allowed", "reranking_allowed"):
        if cs[k] is not False:
            fail(f"controlled_storage.{k} must be false")

    rs = d["reference_storage"]
    if rs["backend"] != "EXACT_IN_MEMORY_DENSE":
        fail("reference_storage.backend must be EXACT_IN_MEMORY_DENSE")
    if rs["approximate_search"] is not False:
        fail("reference_storage.approximate_search must be false")
    if rs.get("candidate_pruning") is not False:
        fail("reference_storage.candidate_pruning must be false")

    ident = d["identity"]
    if "uuid5" not in ident["point_id_source"]:
        fail("identity.point_id_source must reference uuid5")
    if ident["long_unit_id_preserved"] is not True:
        fail("identity.long_unit_id_preserved must be true")

    accept = d["acceptance"]
    if float(accept["top_1_overlap"]) != 1.0:
        fail("top_1_overlap must be 1.0")
    if float(accept["top_5_overlap"]) != 1.0:
        fail("top_5_overlap must be 1.0")

    cls = d["classification"]
    if cls["if_pass"] != "SEMANTICALLY_NEUTRAL_FOR_CONTROLLED_TRACK":
        fail("classification.if_pass wrong")
    if cls["if_fail"] != "STORAGE_SUBSTITUTION_BLOCKED":
        fail("classification.if_fail wrong")

    print(f"[OK] {PATH}")
    print("[OK] Qdrant dense-only, exact reference, uuid5 point ids")


if __name__ == "__main__":
    main()