"""Validate the Step 89 retrieval evaluation execution contract."""

from __future__ import annotations

from pathlib import Path

import yaml

PATH = Path("baseline_audit/pilot_evaluation_execution_contract.yaml")

FROZEN_JUDGMENT_HASH = "717d981d0f10ecffec81ed026e2e535e8c8434d8f2917d38919596e238ea76f6"
FROZEN_EVALUATION_HASH = "7a1580ba5f9f9e276e8d621cf0ed321d522af61bcf69e53086f2295ff66ba592"

FORBIDDEN_EXECUTION_FLAGS = (
    "query_embedding_enabled",
    "retrieval_enabled",
    "qdrant_client_enabled",
    "statistical_significance_testing_enabled",
    "architecture_comparison_enabled",
    "generation_enabled",
    "answer_evaluation_enabled",
    "llm_as_judge_enabled",
    "network_access_enabled",
    "hosted_requests_enabled",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "contract must be a mapping")
    _require(
        data.get("artifact_type") == "pilot_evaluation_execution_contract",
        "artifact_type changed",
    )
    contract = data["contract"]
    _require(
        contract["id"] == "infrastructure_pilot_bounded_retrieval_evaluation_execution",
        "contract.id changed",
    )
    _require(contract["version"] == "0.1.0", "contract.version must be 0.1.0")
    _require(
        contract["status"] == "APPROVED_FOR_BOUNDED_EXECUTION",
        "contract.status must be APPROVED_FOR_BOUNDED_EXECUTION",
    )

    source = data["source"]
    _require(source["expected_query_count"] == 6, "expected_query_count must be 6")
    _require(
        source["expected_ranked_document_count"] == 72,
        "expected_ranked_document_count must be 72",
    )
    _require(
        source["expected_judgment_count"] == 12,
        "expected_judgment_count must be 12",
    )

    run = data["run"]
    _require(run["retrieval_only"] is True, "run.retrieval_only must be true")
    _require(
        run["final_generation_enabled"] is False,
        "final_generation_enabled must be false",
    )
    _require(
        run["answer_evaluation_enabled"] is False,
        "answer_evaluation_enabled must be false",
    )
    _require(
        run["llm_as_judge_enabled"] is False,
        "llm_as_judge_enabled must be false",
    )

    identity = data["identity"]
    _require(
        identity["judgment_plan_hash"] == FROZEN_JUDGMENT_HASH,
        "judgment_plan_hash must match Step 88",
    )
    _require(
        identity["evaluation_plan_hash"] == FROZEN_EVALUATION_HASH,
        "evaluation_plan_hash must match Step 88",
    )

    ev = data["evaluation"]
    _require(ev["unit"] == "canonical_document", "unit must be canonical_document")
    _require(ev["retrieval_method"] == "hybrid_rrf", "retrieval_method must be hybrid_rrf")
    _require(list(ev["cutoffs"]) == [1, 5, 10, 100], "cutoffs must be [1, 5, 10, 100]")
    _require(
        ev["precision_denominator_policy"] == "fixed_cutoff",
        "precision_denominator_policy must be fixed_cutoff",
    )
    _require(
        ev["empty_judgment_policy"] == "reject_case",
        "empty_judgment_policy must be reject_case",
    )
    metrics = set(ev["metric_names"])
    _require(
        metrics
        == {
            "hit_at_k",
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "ndcg_at_k",
        },
        "metric_names mismatch",
    )
    agg = ev["aggregation"]
    _require(agg["method"] == "macro_average", "aggregation.method must be macro_average")
    _require(agg["query_weight"] == "equal", "aggregation.query_weight must be equal")

    exec_ = data["execution"]
    for flag in (
        "runtime_judgment_loading_enabled",
        "retrieval_case_construction_enabled",
        "evaluation_enabled",
        "metric_computation_enabled",
        "aggregate_metric_computation_enabled",
        "persistence_enabled",
    ):
        _require(exec_[flag] is True, "execution." + flag + " must be true")
    for flag in FORBIDDEN_EXECUTION_FLAGS:
        _require(exec_[flag] is False, "execution." + flag + " must be false")

    counts = data["expected_counts"]
    _require(counts["retrieval_case_count"] == 6, "retrieval_case_count must be 6")
    _require(counts["query_metric_rows"] == 24, "query_metric_rows must be 24")
    _require(counts["aggregate_metric_rows"] == 4, "aggregate_metric_rows must be 4")

    pub = data["publishability"]
    _require(pub["publishable_result"] is False, "publishable_result must be false")
    _require(
        pub["metric_values_for_infrastructure_validation_only"] is True,
        "metric_values_for_infrastructure_validation_only must be true",
    )

    print("[OK] Pilot evaluation execution contract")
    print("  id:", contract["id"])
    print("  version:", contract["version"])
    print("  queries:", source["expected_query_count"])
    print("  judgments:", source["expected_judgment_count"])
    print("  cutoffs:", list(ev["cutoffs"]))
    print("  metrics:", len(metrics))
    print("  evaluation enabled:", exec_["evaluation_enabled"])
    print("  retrieval enabled:", exec_["retrieval_enabled"])
    print("  generation enabled:", exec_["generation_enabled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())