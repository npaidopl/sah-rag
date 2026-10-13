"""Validate the Step 92 external canonical suite contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/external_canonical_suite_contract.yaml")

DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")

FORBIDDEN_EXECUTION = (
    "adapter_executed",
    "parser_executed",
    "chunker_executed",
    "tokenizer_loaded",
    "model_loaded",
    "embedding_executed",
    "qdrant_executed",
    "retrieval_executed",
    "evaluation_executed",
    "generation_executed",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "contract must be a mapping")
    _require(
        data.get("artifact_type") == "external_canonical_suite_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "canonical_rag_dataset_suite_registration",
        "contract.id changed",
    )
    _require(
        contract["status"] == "APPROVED_FOR_READ_ONLY_REGISTRATION",
        "contract.status changed",
    )

    suite = data["suite"]
    _require(
        suite["repository_url"]
        == "https://github.com/npaidopl/canonical-rag-dataset-suite",
        "repository_url changed",
    )
    _require(
        len(suite["repository_commit"]) == 40,
        "repository_commit must be 40-char SHA",
    )
    _require(
        suite["expected_collection_fingerprint"]
        == "5f577978b6ab7a4ee3fef29b68db29f95b37261f103bb6cfd5fd7094c70a2aa4",
        "collection fingerprint changed",
    )
    _require(
        suite["expected_document_count"] == 8879,
        "expected_document_count must be 8879",
    )
    _require(
        suite["expected_example_count"] == 29718,
        "expected_example_count must be 29718",
    )

    ds = data["datasets"]
    _require(tuple(ds["expected"]) == DATASETS, "dataset set changed")
    _require(ds["expected_count"] == 7, "expected_count must be 7")

    reg = data["registration"]
    _require(reg["read_only"] is True, "read_only must be true")
    for op in (
        "load_tokenizer",
        "load_model",
        "create_embeddings",
        "execute_qdrant",
        "execute_retrieval",
        "execute_evaluation",
        "execute_generation",
        "execute_external_adapter",
        "modify_external_project",
        "rewrite_jsonl",
        "generate_chunks",
    ):
        _require(op in reg["prohibited_operations"], "must prohibit " + op)

    integrity = data["integrity"]
    for flag in (
        "source_files_must_remain_unchanged",
        "document_ids_unique",
        "query_ids_unique",
        "query_document_references_resolve",
        "relevant_document_references_resolve",
        "canonical_records_must_validate",
        "validation_reports_must_pass",
    ):
        _require(integrity[flag] is True, "integrity." + flag + " must be true")

    exec_ = data["execution"]
    for flag in FORBIDDEN_EXECUTION:
        _require(exec_[flag] is False, "execution." + flag + " must be false")

    pub = data["publishability"]
    _require(
        pub["suite_registration_is_benchmark_result"] is False,
        "suite_registration_is_benchmark_result must be false",
    )

    print("[OK] External canonical suite contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  datasets:", len(ds["expected"]))
    print("  expected documents:", suite["expected_document_count"])
    print("  expected examples:", suite["expected_example_count"])
    print("  read only:", reg["read_only"])
    print("  adapter execution:", exec_["adapter_executed"])
    print("  model loading:", exec_["model_loaded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())