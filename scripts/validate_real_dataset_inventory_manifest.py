"""Validate the Step 91 real dataset inventory manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/real_dataset_inventory_manifest.yaml")

EXPECTED_ID = "seven-real-datasets-read-only-inventory-v1"
ALLOWED_STATUSES = (
    "PLANNED",
    "SUCCEEDED",
    "SUCCEEDED_WITH_BLOCKERS",
    "INTERRUPTED",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "manifest must be a mapping")
    _require(
        data.get("artifact_type") == "real_dataset_inventory_manifest",
        "artifact_type changed",
    )

    execution = data["execution"]
    _require(execution["execution_id"] == EXPECTED_ID, "execution_id changed")
    status = execution["status"]
    _require(status in ALLOWED_STATUSES, "status invalid")

    approval = data["approval"]
    for flag in (
        "path_discovery_approved",
        "read_only_inventory_approved",
        "checksum_computation_approved",
        "structural_sizing_approved",
    ):
        _require(approval[flag] is True, "approval." + flag + " must be true")
    for flag in (
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

    plan = data["plan"]
    _require(plan["expected_dataset_count"] == 7, "expected_dataset_count must be 7")
    _require(
        tuple(plan["expected_dataset_ids"])
        == (
            "ruler",
            "loong",
            "long_bench_v2",
            "lara",
            "greek_bar_bench",
            "cuad",
            "clerk",
        ),
        "expected_dataset_ids changed",
    )

    results = data["results"]
    if status == "PLANNED":
        _require(
            results["inventory_registry_created"] is False,
            "PLANNED must have registry_created false",
        )
    else:
        _require(
            results["inventory_registry_created"] is True,
            "non-PLANNED must have registry_created true",
        )

    for flag in (
        "parser_executed",
        "chunker_executed",
        "tokenizer_loaded",
        "model_loaded",
        "embedding_executed",
        "retrieval_executed",
        "evaluation_executed",
        "generation_executed",
    ):
        _require(results[flag] is False, "results." + flag + " must be false")

    next_gate = data["next_gate"]
    _require(
        next_gate["required_step"] == "model_tokenizer_sizing_plan",
        "next_gate.required_step changed",
    )
    _require(next_gate["approved"] is False, "next_gate.approved must be false")

    print("[OK] Real dataset inventory manifest")
    print("  status:", status)
    print("  resolved datasets:", results["resolved_dataset_count"])
    print("  inventoried datasets:", results["inventoried_dataset_count"])
    print("  blocked datasets:", results["blocked_dataset_count"])
    print("  source files unchanged:", results["source_files_unchanged"])
    print("  parser executed:", results["parser_executed"])
    print("  tokenizer loaded:", results["tokenizer_loaded"])
    print("  model loaded:", results["model_loaded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())