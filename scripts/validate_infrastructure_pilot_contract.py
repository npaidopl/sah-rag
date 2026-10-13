"""Validate the infrastructure-only pilot contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT = Path("baseline_audit/infrastructure_pilot_contract.yaml")
MATRIX = Path("baseline_audit/architecture_environment_matrix.yaml")
PORTFOLIO = Path("baseline_audit/architecture_portfolio_dispositions.yaml")

EXPECTED_INCLUDED = {"longrag", "raptor", "lightrag", "pathrag"}
EXPECTED_EXCLUDED = {
    "hipporag2", "disretrieval", "memorag", "kag", "metadata_rag", "hikey",
}
EXPECTED_STAGES = [
    "preflight", "input_snapshot", "architecture_preprocessing",
    "architecture_indexing", "architecture_retrieval",
    "canonical_document_expansion", "common_evaluation",
    "artifact_finalization",
]
PATHRAG_LABEL = "PathRAG paper-derived architectural comparator"


def main() -> int:
    for p in (CONTRACT, MATRIX, PORTFOLIO):
        if not p.is_file():
            raise SystemExit(f"missing: {p}")

    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    portfolio = yaml.safe_load(PORTFOLIO.read_text(encoding="utf-8"))

    c = data["contract"]
    if c["id"] != "wave1_infrastructure_only_pilot":
        raise SystemExit("contract.id changed")
    if c["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if c["status"] != "FROZEN_FOR_PLANNING":
        raise SystemExit("contract.status must be FROZEN_FOR_PLANNING")

    scope = data["scope"]
    if set(scope["architectures"]) != EXPECTED_INCLUDED:
        raise SystemExit("included architecture set changed")
    if set(scope["excluded_architectures"]) != EXPECTED_EXCLUDED:
        raise SystemExit("excluded architecture set changed")
    if scope["publishable_result"] is not False:
        raise SystemExit("publishable_result must be false")
    if scope["included_in_final_comparison"] is not False:
        raise SystemExit("included_in_final_comparison must be false")
    if scope["retrieval_effectiveness_claimed"] is not False:
        raise SystemExit("retrieval_effectiveness_claimed must be false")

    # Cross-check: all included must be READY_FOR_DATA in portfolio
    for aid in scope["architectures"]:
        assert (
            portfolio["architectures"][aid]["operational_status"] == "READY_FOR_DATA"
        ), f"{aid}: portfolio status not READY_FOR_DATA"

    # Cross-check: all included must be pilot_eligible in matrix
    for aid in scope["architectures"]:
        assert matrix["architectures"][aid]["pilot_eligible"] is True, (
            f"{aid}: not pilot_eligible in matrix"
        )

    # HippoRAG 2 excluded for remote GPU
    hippo = scope["excluded_architectures"]["hipporag2"]
    if hippo["reason"] != "REQUIRES_REMOTE_GPU":
        raise SystemExit("hipporag2 must be excluded for REQUIRES_REMOTE_GPU")

    # PathRAG boundary
    pathrag = data["architecture_constraints"]["pathrag"]
    if pathrag["foreign_repository_execution_allowed"] is not False:
        raise SystemExit("pathrag foreign execution must be false")
    if pathrag["required_execution_label"] != PATHRAG_LABEL:
        raise SystemExit("pathrag required_execution_label changed")

    ds = data["dataset"]
    if ds["dataset_id"] != "infrastructure-pilot-synthetic-v1":
        raise SystemExit("dataset_id changed")
    counts = ds["expected_counts"]
    if counts["document_count"] != 12:
        raise SystemExit("document_count must be 12")
    if counts["chunk_count"] != 24:
        raise SystemExit("chunk_count must be 24")
    if counts["query_count"] != 6:
        raise SystemExit("query_count must be 6")
    if counts["judgment_count"] != 12:
        raise SystemExit("judgment_count must be 12")
    if counts["duplicate_content_pair_count"] != 6:
        raise SystemExit("duplicate_content_pair_count must be 6")

    ex = data["execution"]
    if ex["pilot_execution_enabled"] is not False:
        raise SystemExit("pilot_execution_enabled must be false")
    if ex["baseline_execution_enabled"] is not False:
        raise SystemExit("baseline_execution_enabled must be false")
    if ex["planning_enabled"] is not True:
        raise SystemExit("planning_enabled must be true")

    rb = data["retrieval_boundary"]
    for k in (
        "final_generation_enabled", "answer_evaluation_enabled",
        "llm_as_judge_enabled",
    ):
        if rb[k] is not False:
            raise SystemExit(f"retrieval_boundary.{k} must be false")
    if rb["generated_answers_prohibited"] is not True:
        raise SystemExit("generated_answers_prohibited must be true")
    if rb["architecture_specific_metric_overrides_prohibited"] is not True:
        raise SystemExit("architecture_specific_metric_overrides_prohibited must be true")

    if data["evaluation"]["execution_enabled"] is not False:
        raise SystemExit("evaluation.execution_enabled must be false")

    if data["planning"]["required_stages"] != EXPECTED_STAGES:
        raise SystemExit("required_stages changed")
    if data["planning"]["all_stage_execution_enabled"] is not False:
        raise SystemExit("all_stage_execution_enabled must be false")
    if data["planning"]["architecture_order"] != [
        "longrag", "raptor", "lightrag", "pathrag",
    ]:
        raise SystemExit("architecture_order changed")

    hp = data["host_policy"]
    if hp["cpu_only_ml_execution"] is not True:
        raise SystemExit("cpu_only_ml_execution must be true")
    for k in (
        "cuda_required", "docker_required",
        "hosted_model_calls_allowed", "paid_requests_allowed", "secrets_required",
    ):
        if hp[k] is not False:
            raise SystemExit(f"host_policy.{k} must be false")

    if not data["versioning"]["requires_version_bump"]:
        raise SystemExit("requires_version_bump must not be empty")

    print("[OK] Infrastructure-only pilot contract")
    print(f"  id: {c['id']}")
    print(f"  version: {c['version']}")
    print(f"  architectures: {len(scope['architectures'])}")
    print(f"  excluded architectures: {len(scope['excluded_architectures'])}")
    print(f"  dataset: {ds['dataset_id']}")
    print(f"  documents: {counts['document_count']}")
    print(f"  chunks: {counts['chunk_count']}")
    print(f"  queries: {counts['query_count']}")
    print(f"  judgments: {counts['judgment_count']}")
    print(f"  stages per architecture: {len(EXPECTED_STAGES)}")
    print(f"  execution enabled: {ex['pilot_execution_enabled']}")
    print(f"  evaluation enabled: {data['evaluation']['execution_enabled']}")
    print(f"  generation enabled: {rb['final_generation_enabled']}")
    print(f"  publishable result: {scope['publishable_result']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
