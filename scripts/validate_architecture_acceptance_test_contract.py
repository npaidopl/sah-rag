"""Validate the Step 94 common acceptance test contract."""
from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/architecture_acceptance_test_contract.yaml")


def _require(c: bool, m: str) -> None:
    if not c:
        raise SystemExit(m)


def main() -> int:
    d = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(d.get("artifact_type") == "architecture_acceptance_test_contract",
             "artifact_type changed")
    c = d["contract"]
    _require(c["id"] == "common_architecture_synthetic_acceptance",
             "contract.id changed")
    _require(c["status"] == "FROZEN_FOR_IMPLEMENTATION",
             "contract.status changed")

    f = d["fixture"]
    _require(f["documents"] == 12, "documents must be 12")
    _require(f["queries"] == 6, "queries must be 6")
    _require(f["relevant_documents_per_query"] == 2,
             "relevant_documents_per_query must be 2")

    stages = d["required_stages"]
    for s in (
        "canonical_input_load","architecture_preprocessing",
        "architecture_state_creation","architecture_state_persistence",
        "architecture_state_reopen","retrieval",
        "canonical_identity_expansion","common_evaluation_case_construction",
    ):
        _require(s in stages, f"required_stage {s} missing")

    _require(d["generation"]["enabled"] is False, "generation must be disabled")

    a = d["acceptance"]
    for flag in (
        "canonical_input_accepted","persistent_state_created",
        "persistent_state_reopened","retrieval_completed_for_every_query",
        "ranked_results_nonempty","canonical_ids_recovered",
        "duplicate_document_identities_preserved",
        "ground_truth_not_used_by_retriever",
        "deterministic_or_classified",
        "retrieval_artifacts_checksum_bound",
        "source_artifacts_unchanged",
    ):
        _require(a[flag] is True, "acceptance." + flag + " must be true")
    _require(a["query_count"] == 6, "query_count must be 6")

    for p in ("generated_answers","LLM_judge","architecture_comparison",
              "publishable_claims","test_ground_truth_during_retrieval"):
        _require(p in d["prohibited"], f"prohibited {p} missing")

    _require(d["common_gates_may_not_be_removed"] is True,
             "common_gates_may_not_be_removed must be true")

    print("[OK] Common architecture synthetic acceptance contract")
    print("  fixture documents:", f["documents"])
    print("  fixture queries:", f["queries"])
    print("  required stages:", len(stages))
    print("  generation enabled:", d["generation"]["enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())