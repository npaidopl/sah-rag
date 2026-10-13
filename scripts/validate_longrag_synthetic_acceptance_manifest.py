"""Validate baseline_audit/longrag_synthetic_acceptance_manifest.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/longrag_synthetic_acceptance_manifest.yaml")

ALLOWED_STATUS = {"PLANNED", "RUNNING", "PASSED", "FAILED"}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    if not PATH.exists():
        fail(f"missing: {PATH}")

    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))

    if d["artifact_type"] != "longrag_synthetic_acceptance_manifest":
        fail("wrong artifact_type")

    ex = d["execution"]
    if ex["execution_id"] != "longrag-controlled-synthetic-acceptance-v1":
        fail("wrong execution_id")
    if ex["status"] not in ALLOWED_STATUS:
        fail(f"status must be in {ALLOWED_STATUS}")

    appr = d["approval"]
    required_true = {
        "long_unit_construction_approved", "model_loading_approved",
        "dense_embedding_approved", "reference_index_approved",
        "qdrant_collection_approved", "reference_retrieval_approved",
        "qdrant_retrieval_approved", "equivalence_comparison_approved",
        "canonical_expansion_approved", "evaluation_case_construction_approved",
    }
    required_false = {
        "native_embedder_execution_approved", "sparse_embedding_approved",
        "hybrid_retrieval_approved", "llm_execution_approved",
        "metric_computation_approved", "generation_approved",
    }
    for k in required_true:
        if appr.get(k) is not True:
            fail(f"approval.{k} must be true")
    for k in required_false:
        if appr.get(k) is not False:
            fail(f"approval.{k} must be false")

    ng = d["next_gate"]
    if ng["required_step"] != "longrag_readiness_declaration":
        fail("next_gate.required_step wrong")

    print(f"[OK] {PATH}")
    print(f"[OK] status={ex['status']}")


if __name__ == "__main__":
    main()