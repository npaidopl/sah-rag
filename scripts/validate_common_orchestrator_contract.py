"""Validate the common orchestrator contract."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONTRACT_PATH = Path("baseline_audit/common_orchestrator_contract.yaml")
EXPECTED_ARCHITECTURES = {"longrag", "raptor", "hipporag2", "lightrag", "pathrag"}
EXPECTED_STAGES = [
    "preflight",
    "input_snapshot",
    "architecture_preprocessing",
    "architecture_indexing",
    "architecture_retrieval",
    "canonical_document_expansion",
    "common_evaluation",
    "artifact_finalization",
]


def main() -> int:
    if not CONTRACT_PATH.is_file():
        raise SystemExit(f"missing contract: {CONTRACT_PATH}")
    data = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))

    c = data["contract"]
    if c["id"] != "wave1_common_orchestrator":
        raise SystemExit("contract.id changed")
    if c["version"] != "0.1.0":
        raise SystemExit("contract.version must be 0.1.0")
    if c["status"] != "FROZEN_FOR_IMPLEMENTATION":
        raise SystemExit("contract.status must be FROZEN_FOR_IMPLEMENTATION")

    scope = data["scope"]
    if set(scope["architectures"]) != EXPECTED_ARCHITECTURES:
        raise SystemExit("scope.architectures changed")
    if scope["mode"] != "dry_run_planning":
        raise SystemExit("scope.mode must be dry_run_planning")
    for k in (
        "architecture_execution_enabled",
        "dataset_mutation_enabled",
        "final_generation_enabled",
        "evaluation_execution_enabled",
    ):
        if scope[k] is not False:
            raise SystemExit(f"scope.{k} must be false")

    if data["planning"]["stages"] != EXPECTED_STAGES:
        raise SystemExit("planning.stages changed")
    if data["planning"]["no_execution"] is not True:
        raise SystemExit("planning.no_execution must be true")

    safety = data["safety"]
    for k in (
        "architecture_execution_enabled",
        "foreign_code_execution_enabled",
        "model_download_enabled",
        "network_access_required",
        "final_generation_enabled",
        "evaluation_execution_enabled",
        "secrets_in_artifacts",
    ):
        if safety[k] is not False:
            raise SystemExit(f"safety.{k} must be false")

    arch = data["architecture_planning"]
    if arch["pathrag"]["execution_label"] != (
        "PathRAG paper-derived architectural comparator"
    ):
        raise SystemExit("PathRAG label changed")
    if arch["pathrag"]["implementation_boundary"] != "paper_derived_reimplementation":
        raise SystemExit("PathRAG boundary changed")
    if arch["pathrag"]["foreign_execution_allowed"] is not False:
        raise SystemExit("PathRAG foreign execution must be disallowed")

    print("[OK] Common orchestrator contract")
    print(f"  id: {c['id']}")
    print(f"  version: {c['version']}")
    print(f"  mode: {scope['mode']}")
    print(f"  architectures: {len(scope['architectures'])}")
    print(f"  stages: {len(data['planning']['stages'])}")
    print(f"  architecture execution: {scope['architecture_execution_enabled']}")
    print(f"  evaluation execution: {scope['evaluation_execution_enabled']}")
    print(f"  final generation: {scope['final_generation_enabled']}")
    print(f"  PathRAG foreign execution: {arch['pathrag']['foreign_execution_allowed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
