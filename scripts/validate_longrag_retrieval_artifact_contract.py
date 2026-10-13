"""Validate baseline_audit/longrag_retrieval_artifact_contract.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PATH = Path("baseline_audit/longrag_retrieval_artifact_contract.yaml")

REQUIRED_TOP_KEYS = {
    "artifact_type", "artifact_version", "contract", "field_verification",
    "long_units", "embeddings", "reference_rankings", "qdrant_rankings",
    "ranked_documents", "evaluation_cases", "query_projection",
    "constraints", "prohibited_fields",
}

REQUIRED_EMBEDDING_FIELDS = {
    "long_unit_id", "condition_id", "dataset_id", "construction_policy",
    "target_token_size", "maximum_model_input_tokens", "unit_order",
    "rendered_token_count", "special_token_count", "actual_model_token_count",
    "was_truncated", "member_doc_ids", "member_parent_ids", "member_chunk_ids",
    "model_id", "model_revision", "tokenizer_id", "embedding_config_hash",
    "dense_vector", "dense_dimension", "dense_precision", "dense_normalized",
    "source_unit_sha256", "source_units_file_sha256", "adapter_id",
    "adapter_version", "source_architecture", "source_architecture_commit",
    "interface_validation", "paper_faithful_result", "metadata",
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

    if d["artifact_type"] != "longrag_retrieval_artifact_contract":
        fail("wrong artifact_type")

    if d["contract"]["status"] != "FROZEN_FOR_EXECUTION":
        fail("contract.status must be FROZEN_FOR_EXECUTION")

    fields = set(d["embeddings"]["required_fields"])
    missing_fields = REQUIRED_EMBEDDING_FIELDS - fields
    if missing_fields:
        fail(f"embeddings.required_fields missing: {sorted(missing_fields)}")

    qp = d["query_projection"]
    for forbidden in ("doc_ids", "relevant_doc_ids", "answers", "ground_truth"):
        if forbidden not in qp["prohibited_fields"]:
            fail(f"query_projection must prohibit {forbidden}")
    for required in ("query_id", "query_text"):
        if required not in qp["allowed_fields"]:
            fail(f"query_projection must allow {required}")

    constraints = d["constraints"]
    if constraints["ranks_start_at_one"] is not True:
        fail("ranks_start_at_one must be true")
    if constraints["score_direction"] != "HIGHER_IS_BETTER":
        fail("score_direction must be HIGHER_IS_BETTER")

    print(f"[OK] {PATH}")
    print(f"[OK] {len(fields)} embedding fields frozen; query projection guarded")


if __name__ == "__main__":
    main()