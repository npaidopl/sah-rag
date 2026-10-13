"""Validate the Step 92 external canonical suite manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/external_canonical_suite_manifest.yaml")

FORBIDDEN_RESULTS = (
    "adapter_executed",
    "parser_executed",
    "chunker_executed",
    "tokenizer_loaded",
    "model_loaded",
    "embedding_executed",
    "retrieval_executed",
    "evaluation_executed",
    "generation_executed",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(
        data.get("artifact_type") == "external_canonical_suite_manifest",
        "artifact_type changed",
    )
    execution = data["execution"]
    status = execution["status"]
    _require(
        status in ("PLANNED", "SUCCEEDED", "INTERRUPTED"),
        "status must be valid",
    )

    approval = data["approval"]
    for flag in (
        "external_root_registration_approved",
        "manifest_inspection_approved",
        "canonical_hashing_approved",
        "streaming_validation_approved",
        "corpus_scope_audit_approved",
        "ground_truth_projection_audit_approved",
    ):
        _require(approval[flag] is True, "approval." + flag + " must be true")
    for flag in (
        "adapter_execution_approved",
        "parser_execution_approved",
        "chunker_execution_approved",
        "tokenizer_loading_approved",
        "model_loading_approved",
        "embedding_execution_approved",
        "retrieval_execution_approved",
        "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        _require(approval[flag] is False, "approval." + flag + " must be false")

    r = data["results"]
    if status == "PLANNED":
        _require(
            r["external_root_resolved"] is False,
            "PLANNED must have external_root_resolved false",
        )
    else:
        _require(
            r["external_root_resolved"] is True,
            "SUCCEEDED must have external_root_resolved true",
        )
        _require(r["dataset_count"] == 7, "dataset_count must be 7")
        _require(r["document_count"] == 8879, "document_count must be 8879")
        _require(r["example_count"] == 29718, "example_count must be 29718")
        _require(r["manifests_valid"] is True, "manifests_valid must be true")
        _require(r["validations_passed"] is True, "validations_passed must be true")
        _require(
            r["collection_fingerprint_valid"] is True,
            "collection_fingerprint_valid must be true",
        )
        _require(
            r["registration_registry_created"] is True,
            "registration_registry_created must be true",
        )
        _require(r["source_files_unchanged"] is True, "source_files_unchanged must be true")
        _require(
            r["corpus_scope_contract_created"] is True,
            "corpus_scope_contract_created must be true",
        )
        _require(
            r["ground_truth_projection_contract_created"] is True,
            "ground_truth_projection_contract_created must be true",
        )

    for flag in FORBIDDEN_RESULTS:
        _require(r[flag] is False, "results." + flag + " must be false")

    ng = data["next_gate"]
    _require(
        ng["required_step"] == "fair_benchmark_protocol_and_fidelity_matrix",
        "next_gate.required_step changed",
    )
    _require(ng["approved"] is False, "next_gate.approved must be false")

    print("[OK] External canonical suite manifest")
    print("  status:", status)
    print("  datasets:", r["dataset_count"])
    print("  documents:", r["document_count"])
    print("  examples:", r["example_count"])
    print("  fingerprint valid:", r["collection_fingerprint_valid"])
    print("  source files unchanged:", r["source_files_unchanged"])
    print("  adapter executed:", r["adapter_executed"])
    print("  model loaded:", r["model_loaded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())