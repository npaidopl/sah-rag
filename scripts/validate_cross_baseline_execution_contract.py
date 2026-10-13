"""Validate the cross-baseline execution contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT_PATH = Path("baseline_audit/cross_baseline_execution_contract.yaml")

EXPECTED_ARCHITECTURES = {
    "longrag", "raptor", "hipporag2", "lightrag", "pathrag",
}

EXPECTED_IDENTITY_FIELDS = (
    "dataset_id",
    "dataset_revision",
    "split",
    "architecture_id",
    "adapter_id",
    "adapter_version",
    "architecture_configuration_hash",
    "canonical_input_manifest_sha256",
    "random_seed",
    "code_revision",
)

EXPECTED_STATUS_TRANSITIONS = {
    "PLANNED": {"RUNNING", "CANCELLED"},
    "RUNNING": {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"},
    "INTERRUPTED": {"RUNNING", "CANCELLED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

EXPECTED_METRICS = {
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "hit_at_k",
}


def load() -> dict:
    if not CONTRACT_PATH.is_file():
        raise SystemExit(f"missing contract: {CONTRACT_PATH}")
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def main() -> int:
    data = load()

    contract = data["contract"]
    if contract["id"] != "wave1_cross_baseline_execution":
        raise SystemExit("contract.id changed")
    if contract["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if contract["status"] != "FROZEN_FOR_IMPLEMENTATION":
        raise SystemExit("contract.status must be FROZEN_FOR_IMPLEMENTATION")

    scope = data["scope"]
    if set(scope["architectures"]) != EXPECTED_ARCHITECTURES:
        raise SystemExit("scope.architectures must be exactly Wave 1")
    if scope["execution_level"] != "retrieval_only":
        raise SystemExit("scope.execution_level must be retrieval_only")
    if scope["final_generation_enabled"] is not False:
        raise SystemExit("scope.final_generation_enabled must be false")
    if scope["answer_evaluation_enabled"] is not False:
        raise SystemExit("scope.answer_evaluation_enabled must be false")

    pb = scope["pathrag_boundary"]
    if pb["foreign_repository_execution_allowed"] is not False:
        raise SystemExit("PathRAG foreign execution must remain disallowed")
    if pb["comparator_only"] is not True:
        raise SystemExit("PathRAG comparator_only must be true")

    cell = data["experiment_cell"]
    if tuple(cell["identity_fields"]) != EXPECTED_IDENTITY_FIELDS:
        raise SystemExit("experiment_cell.identity_fields changed")
    if cell["identity_algorithm"] != "sha256_canonical_json":
        raise SystemExit("experiment_cell.identity_algorithm changed")

    manifest = data["architecture_execution_manifest"]
    if set(manifest["execution_status_values"]) != set(EXPECTED_STATUS_TRANSITIONS):
        raise SystemExit("execution_status_values changed")
    if set(manifest["terminal_status_values"]) != {
        "SUCCEEDED", "FAILED", "CANCELLED"
    }:
        raise SystemExit("terminal_status_values changed")
    for status, allowed in EXPECTED_STATUS_TRANSITIONS.items():
        got = set(manifest["status_transitions"][status])
        if got != allowed:
            raise SystemExit(f"status transition for {status} changed")

    rb = data["retrieval_boundary"]
    for k in ("final_generation_enabled", "qa_model_enabled", "llm_as_judge_enabled"):
        if rb[k] is not False:
            raise SystemExit(f"retrieval_boundary.{k} must be false")

    evaluation = data["evaluation"]
    if evaluation["level"] != "document":
        raise SystemExit("evaluation.level must be document")
    if set(evaluation["metrics"]) != EXPECTED_METRICS:
        raise SystemExit("evaluation.metrics changed")
    if evaluation["cutoffs"] != [1, 5, 10, 100]:
        raise SystemExit("evaluation.cutoffs changed")

    resume = data["resume_semantics"]
    if resume["resume_allowed_from"] != ["INTERRUPTED"]:
        raise SystemExit("resume_allowed_from changed")
    if set(resume["resume_prohibited_from"]) != {
        "SUCCEEDED", "FAILED", "CANCELLED"
    }:
        raise SystemExit("resume_prohibited_from changed")

    imm = data["immutability"]
    for k in (
        "succeeded_runs_immutable",
        "overwrite_succeeded_run_prohibited",
        "rerun_requires_new_experiment_cell_id",
    ):
        if imm[k] is not True:
            raise SystemExit(f"immutability.{k} must be true")

    provenance = set(data["provenance"]["required"])
    for req in (
        "project_code_revision",
        "adapter_version",
        "architecture_source_revision",
        "dataset_revision",
        "configuration_hash",
        "input_manifest_sha256",
        "execution_environment_id",
        "random_seed",
    ):
        if req not in provenance:
            raise SystemExit(f"provenance missing: {req}")

    if data["versioning"]["contract_version"] != "0.1.0":
        raise SystemExit("versioning.contract_version changed")

    print("[OK] Cross-baseline execution contract")
    print(f"  id: {contract['id']}")
    print(f"  version: {contract['version']}")
    print(f"  architectures: {len(scope['architectures'])}")
    print("  retrieval boundary: generation disabled")
    print("  PathRAG foreign execution: disallowed")
    print(f"  evaluation cutoffs: {evaluation['cutoffs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
