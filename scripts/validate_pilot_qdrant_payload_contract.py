"""Validate the frozen Qdrant payload contract for the pilot dataset."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_qdrant_payload_contract.yaml")

FROZEN_MODEL_ID = "BAAI/bge-m3"
FROZEN_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FROZEN_DATASET = "infrastructure-pilot-synthetic-v1"

REQUIRED_PAYLOAD_FIELDS = (
    "chunk_id",
    "parent_id",
    "doc_id",
    "dataset",
    "chunk_index",
    "parent_chunk_index",
    "text_normalized",
    "normalized_char_start",
    "normalized_char_end",
    "token_count",
    "model_id",
    "model_revision",
    "embedding_config_hash",
    "embedding_shard_id",
)

NULLABLE_PAYLOAD_FIELDS = ("token_start", "token_end")

PROHIBITED = {
    "generated_answer",
    "reference_answer",
    "answer_text",
    "relevance",
    "relevance_label",
    "judgment",
    "API_key",
    "access_token",
    "secret",
    "credentials",
    "raw_model_weights",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "payload contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_qdrant_payload_contract",
        "artifact_type must be pilot_qdrant_payload_contract",
    )
    _require(data.get("artifact_version") == "1.0", "artifact_version must be 1.0")

    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_qdrant_payload",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")
    _require(
        contract["status"] == "FROZEN_FOR_PLANNING",
        "contract.status must be FROZEN_FOR_PLANNING",
    )

    _require(
        data["dataset"]["dataset_id"] == FROZEN_DATASET,
        "dataset.dataset_id changed",
    )
    _require(
        data["model"]["model_id"] == FROZEN_MODEL_ID,
        "model.model_id must be " + FROZEN_MODEL_ID,
    )
    _require(
        data["model"]["model_revision"] == FROZEN_REVISION,
        "model.model_revision must equal frozen revision",
    )

    impl = data["implementation"]
    _require(
        impl["builder_function"] == "sah_rag.storage.qdrant.build_payload",
        "implementation.builder_function changed",
    )
    _require(impl["frozen"] is True, "implementation.frozen must be true")

    payload = data["payload"]
    _require(payload["schema_version"] == "0.1.0", "payload.schema_version must be 0.1.0")

    required = payload["required_fields"]
    for name in REQUIRED_PAYLOAD_FIELDS:
        _require(name in required, "required_fields missing " + name)

    nullable = payload["nullable_fields"]
    for name in NULLABLE_PAYLOAD_FIELDS:
        _require(name in nullable, "nullable_fields missing " + name)

    # Exact identity values
    _require(
        required["dataset"]["exact_value"] == FROZEN_DATASET,
        "payload dataset exact_value must be " + FROZEN_DATASET,
    )
    _require(
        required["model_id"]["exact_value"] == FROZEN_MODEL_ID,
        "payload model_id exact_value must be " + FROZEN_MODEL_ID,
    )
    _require(
        required["model_revision"]["exact_value"] == FROZEN_REVISION,
        "payload model_revision exact_value must equal frozen revision",
    )
    _require(
        required["embedding_config_hash"]["format"] == "lowercase_sha256",
        "embedding_config_hash format must be lowercase_sha256",
    )

    # Prohibited fields
    prohibited = set(payload["prohibited_fields"])
    missing_prohibited = PROHIBITED - prohibited
    _require(
        not missing_prohibited,
        "prohibited_fields missing: " + ", ".join(sorted(missing_prohibited)),
    )

    # No evaluation-only fields should appear in required
    forbidden_in_required = PROHIBITED & set(required.keys())
    _require(
        not forbidden_in_required,
        "evaluation-only fields appear in required: "
        + ", ".join(sorted(forbidden_in_required)),
    )

    lineage = data["lineage"]
    _require(
        lineage["required_chain"][0] == "point_id",
        "lineage.required_chain must start with point_id",
    )
    _require(
        lineage["canonical_document_coverage_required"] is True,
        "canonical_document_coverage_required must be true",
    )

    spec_notes = payload.get("spec_requested_but_not_produced", {})
    _require(
        "dataset_id" in spec_notes
        and "source_order" in spec_notes
        and "content_sha256" in spec_notes,
        "spec_requested_but_not_produced must document all three spec divergences",
    )

    validation = data["validation"]
    _require(
        validation["payload_field_set_must_match_builder"] is True,
        "payload_field_set_must_match_builder must be true",
    )

    print("[OK] Pilot Qdrant payload contract")
    print("  contract id:", contract["id"])
    print("  version:", contract["version"])
    print("  required fields:", len(REQUIRED_PAYLOAD_FIELDS))
    print("  nullable fields:", len(NULLABLE_PAYLOAD_FIELDS))
    print("  dataset:", data["dataset"]["dataset_id"])
    print("  model:", data["model"]["model_id"])
    print("  model revision:", data["model"]["model_revision"])
    print("  evaluation fields prohibited:", True)
    print("  secret fields prohibited:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())