"""Validate the Step 91 real dataset inventory contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/real_dataset_inventory_contract.yaml")

EXPECTED_DATASETS = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "contract must be a mapping")
    _require(
        data.get("artifact_type") == "real_dataset_inventory_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "seven_real_long_context_datasets_inventory",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")

    expected = tuple(data["datasets"]["expected"])
    _require(expected == EXPECTED_DATASETS, "datasets set changed")
    _require(data["datasets"]["expected_count"] == 7, "expected_count must be 7")
    _require(
        data["datasets"]["exact_set_required"] is True,
        "exact_set_required must be true",
    )

    coll = data["collection"]
    _require(coll["read_only"] is True, "collection.read_only must be true")
    _require(
        coll["filesystem_mutation_allowed"] is False,
        "filesystem_mutation_allowed must be false",
    )
    for op in (
        "load_model_tokenizer",
        "load_embedding_model",
        "create_embeddings",
        "create_qdrant_collection",
        "execute_retrieval",
        "execute_evaluation",
        "execute_generation",
        "parse_documents",
        "generate_chunks",
    ):
        _require(op in coll["prohibited_operations"], "must prohibit " + op)

    _require(
        data["model_input_tokens"]["status"] == "NOT_MEASURED_IN_STEP_91",
        "model_input_tokens.status must be NOT_MEASURED_IN_STEP_91",
    )
    _require(
        data["estimated_model_tokens"]["allowed"] is False,
        "estimated_model_tokens.allowed must be false",
    )
    _require(
        data["statistics"]["percentile_method"] == "nearest_rank",
        "percentile_method must be nearest_rank",
    )

    sizing = data["sizing"]
    _require(
        sizing["estimates_must_be_structural_only"] is True,
        "estimates_must_be_structural_only must be true",
    )
    _require(
        sizing["elapsed_time_estimates_allowed"] is False,
        "elapsed_time_estimates_allowed must be false",
    )

    exec_ = data["execution"]
    for flag in (
        "parser_executed",
        "chunker_executed",
        "tokenizer_loaded",
        "model_loaded",
        "embedding_executed",
        "qdrant_executed",
        "retrieval_executed",
        "evaluation_executed",
        "generation_executed",
    ):
        _require(exec_[flag] is False, "execution." + flag + " must be false")

    pub = data["publishability"]
    _require(pub["publishable_result"] is False, "publishable_result must be false")

    print("[OK] Real dataset inventory contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  datasets:", len(expected))
    print("  read only:", coll["read_only"])
    print("  tokenizer loading:", exec_["tokenizer_loaded"])
    print("  model loading:", exec_["model_loaded"])
    print("  embedding execution:", exec_["embedding_executed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())