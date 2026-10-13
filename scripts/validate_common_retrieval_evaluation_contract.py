"""Validate the common retrieval evaluation contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT_PATH = Path("baseline_audit/common_retrieval_evaluation_contract.yaml")
EXPECTED_ARCHITECTURES = {"longrag", "raptor", "hipporag2", "lightrag", "pathrag"}


def main() -> int:
    if not CONTRACT_PATH.is_file():
        raise SystemExit(f"missing contract: {CONTRACT_PATH}")
    data = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))

    c = data["contract"]
    if c["id"] != "wave1_common_retrieval_evaluation":
        raise SystemExit("contract.id changed")
    if c["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if c["status"] != "FROZEN_FOR_IMPLEMENTATION":
        raise SystemExit("contract.status must be FROZEN_FOR_IMPLEMENTATION")

    scope = data["scope"]
    if scope["evaluation_level"] != "document":
        raise SystemExit("scope.evaluation_level must be document")
    if set(scope["architectures"]) != EXPECTED_ARCHITECTURES:
        raise SystemExit("scope.architectures changed")
    for k in (
        "retrieval_only",
        "final_generation_enabled",
        "answer_evaluation_enabled",
        "llm_as_judge_enabled",
    ):
        if scope[k] is not (k == "retrieval_only"):
            raise SystemExit(f"scope.{k} has wrong polarity")

    if data["cutoffs"] != [1, 5, 10, 100]:
        raise SystemExit("cutoffs changed")

    m = data["metric_definitions"]
    if m["precision_at_k"]["denominator_policy"] != "fixed_cutoff":
        raise SystemExit("precision denominator policy changed")
    if m["reciprocal_rank"]["cutoff_applied"] is not True:
        raise SystemExit("reciprocal_rank must respect the cutoff")
    if m["ndcg_at_k"]["gain_formula"] != "2^relevance - 1":
        raise SystemExit("nDCG gain formula changed")
    if m["ndcg_at_k"]["discount_formula"] != "1 / log2(rank + 1)":
        raise SystemExit("nDCG discount formula changed")
    if m["recall_at_k"]["zero_relevant_policy"] != "reject_case":
        raise SystemExit("recall zero-relevant policy changed")
    if m["ndcg_at_k"]["zero_idcg_policy"] != "reject_case":
        raise SystemExit("nDCG zero-idcg policy changed")

    agg = data["aggregation"]
    if agg["method"] != "macro_average":
        raise SystemExit("aggregation.method must be macro_average")
    if agg["weighting"] != "equal_query_weight":
        raise SystemExit("aggregation.weighting changed")

    safety = data["safety"]
    for k in (
        "final_generation_enabled",
        "answer_evaluation_enabled",
        "llm_as_judge_enabled",
        "generated_answers_in_artifacts",
        "architecture_specific_metric_overrides_allowed",
    ):
        if safety[k] is not False:
            raise SystemExit(f"safety.{k} must be false")
    if safety["retrieval_only"] is not True:
        raise SystemExit("safety.retrieval_only must be true")

    print("[OK] Common retrieval evaluation contract")
    print(f"  id: {c['id']}")
    print(f"  version: {c['version']}")
    print(f"  level: {scope['evaluation_level']}")
    print(f"  cutoffs: {data['cutoffs']}")
    print(f"  aggregation: {agg['method']}")
    print(f"  precision denominator: {m['precision_at_k']['denominator_policy']}")
    print(f"  NDCG gain: {m['ndcg_at_k']['gain_formula']}")
    print(f"  final generation: {safety['final_generation_enabled']}")
    print(f"  architecture overrides: {safety['architecture_specific_metric_overrides_allowed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
