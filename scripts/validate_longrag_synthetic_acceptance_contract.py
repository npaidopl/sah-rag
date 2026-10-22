"""Validate baseline_audit/longrag_synthetic_acceptance_contract.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/longrag_synthetic_acceptance_contract.yaml")

REQUIRED_TOP_KEYS = {
    "artifact_type", "artifact_version", "contract", "dependencies",
    "source", "architecture", "execution", "longrag",
    "controlled_embedding", "reference_backend", "qdrant_backend",
    "acceptance_gates", "publishability",
}

REQUIRED_EXECUTION_TRUE = {
    "canonical_input_load_enabled", "long_unit_construction_enabled",
    "dense_embedding_enabled", "reference_dense_index_enabled",
    "qdrant_index_enabled", "reference_retrieval_enabled",
    "qdrant_retrieval_enabled", "storage_equivalence_enabled",
    "canonical_expansion_enabled", "evaluation_case_construction_enabled",
}
REQUIRED_EXECUTION_FALSE = {
    "sparse_embedding_enabled", "hybrid_retrieval_enabled", "reranking_enabled",
    "llm_execution_enabled", "generation_enabled", "metric_computation_enabled",
}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if not PATH.exists():
        fail(f"missing: {PATH}")

    raw = PATH.read_text(encoding="utf-8")
    if "<actual>" in raw:
        fail("<actual> placeholder remains")

    d = yaml.safe_load(raw)

    missing = REQUIRED_TOP_KEYS - set(d.keys())
    if missing:
        fail(f"missing top-level keys: {sorted(missing)}")

    if d["artifact_type"] != "longrag_synthetic_acceptance_contract":
        fail("wrong artifact_type")
    if d["contract"]["status"] != "APPROVED_FOR_BOUNDED_EXECUTION":
        fail("contract.status must be APPROVED_FOR_BOUNDED_EXECUTION")

    src = d["source"]
    if src["expected_documents"] != 12:
        fail("expected_documents must be 12")
    if src["expected_queries"] != 6:
        fail("expected_queries must be 6")
    if src["expected_relevant_documents_per_query"] != 2:
        fail("expected_relevant_documents_per_query must be 2")

    arch = d["architecture"]
    if arch["architecture_id"] != "longrag":
        fail("architecture_id must be longrag")
    if arch["fidelity_class"] != "CONTROLLED_ADAPTATION":
        fail("fidelity_class must be CONTROLLED_ADAPTATION")
    if arch["controlled_track_eligible"] is not True:
        fail("controlled_track_eligible must be true")
    if arch["native_track_eligible"] is not False:
        fail("native_track_eligible must be false")

    ex = d["execution"]
    for k in REQUIRED_EXECUTION_TRUE:
        if ex.get(k) is not True:
            fail(f"execution.{k} must be true")
    for k in REQUIRED_EXECUTION_FALSE:
        if ex.get(k) is not False:
            fail(f"execution.{k} must be false")

    lr = d["longrag"]
    if lr["grouping_policy"] != "adjacent_segments":
        fail("grouping_policy must be adjacent_segments")
    if lr["target_token_size"] != 256:
        fail("target_token_size must be 256")
    for k in ("long_unit_builder_function", "long_unit_identity_function",
              "canonical_expansion_function"):
        if not lr.get(k, "").startswith("baselines.longrag"):
            fail(f"longrag.{k} must be a baselines.longrag function")

    gates = d["acceptance_gates"]
    if gates["documents_loaded"] != 12:
        fail("gates.documents_loaded must be 12")
    if gates["queries_loaded"] != 6:
        fail("gates.queries_loaded must be 6")
    if gates["storage_equivalence_passed"] is not True:
        fail("gates.storage_equivalence_passed must be true")
    if gates["ground_truth_loaded_during_retrieval"] is not False:
        fail("gates.ground_truth_loaded_during_retrieval must be false")

    pub = d["publishability"]
    if pub["publishable_result"] is not False:
        fail("publishable_result must be false")
    if pub["native_faithful_reproduction"] is not False:
        fail("native_faithful_reproduction must be false")
    if pub["controlled_adaptation"] is not True:
        fail("controlled_adaptation must be true")

    print(f"[OK] {PATH}")
    print("[OK] 12 docs / 6 queries / controlled-only, no placeholders")


if __name__ == "__main__":
    main()