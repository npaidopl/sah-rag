"""Validate the Step 88 pilot retrieval evaluation contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_evaluation_contract.yaml")

EXPECTED_CUTOFFS = [1, 5, 10, 100]
EXPECTED_METRICS = {
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_retrieval_evaluation_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_common_retrieval_evaluation",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")

    source = data["source"]
    _require(source["expected_query_count"] == 6, "expected_query_count must be 6")
    _require(source["expected_judgment_count"] == 12, "expected_judgment_count must be 12")
    _require(
        source["expected_relevant_documents_per_query"] == 2,
        "expected_relevant_documents_per_query must be 2",
    )

    ev = data["evaluation"]
    _require(ev["unit"] == "canonical_document", "unit must be canonical_document")
    _require(
        ev["primary_retrieval_method"] == "hybrid_rrf",
        "primary_retrieval_method must be hybrid_rrf",
    )
    _require(list(ev["cutoffs"]) == EXPECTED_CUTOFFS, "cutoffs must be [1, 5, 10, 100]")
    _require(ev["synthetic_padding_allowed"] is False, "synthetic_padding_allowed must be false")
    _require(
        ev["missing_rank_invention_allowed"] is False,
        "missing_rank_invention_allowed must be false",
    )
    _require(ev["relevance_type"] == "binary", "relevance_type must be binary")

    metrics = set(data["metric_names"])
    _require(metrics == EXPECTED_METRICS, "metric_names mismatch")

    _require(
        data["precision_denominator_policy"] == "fixed_cutoff",
        "precision_denominator_policy must be fixed_cutoff",
    )
    _require(
        data["empty_judgment_policy"] == "reject_case",
        "empty_judgment_policy must be reject_case",
    )

    exec_ = data["execution"]
    _require(exec_["planning_enabled"] is True, "planning_enabled must be true")
    _require(
        exec_["planning_judgment_inspection_allowed"] is True,
        "planning_judgment_inspection_allowed must be true",
    )
    for flag in (
        "runtime_judgment_loading_enabled",
        "evaluation_enabled",
        "metric_computation_enabled",
        "statistical_analysis_enabled",
        "retrieval_enabled",
        "generation_enabled",
        "answer_evaluation_enabled",
        "llm_as_judge_enabled",
    ):
        _require(exec_[flag] is False, "execution." + flag + " must be false")

    print("[OK] Pilot retrieval evaluation contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  queries:", source["expected_query_count"])
    print("  judgments:", source["expected_judgment_count"])
    print("  retrieval method:", ev["primary_retrieval_method"])
    print("  evaluation unit:", ev["unit"])
    print("  cutoffs:", list(ev["cutoffs"]))
    print("  evaluation enabled:", exec_["evaluation_enabled"])
    print("  generation enabled:", exec_["generation_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())