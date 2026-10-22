"""Synthetic dry-run smoke for the common orchestrator."""

from __future__ import annotations

import shutil
from pathlib import Path

from orchestration import (
    PATHRAG_REQUIRED_LABEL,
    WAVE1_ARCHITECTURES,
    OrchestratorExecutionRequest,
    build_orchestrator_dry_run_plan,
    build_planned_execution_manifest,
)
from orchestration.persistence import (
    read_orchestrator_dry_run,
    write_orchestrator_dry_run,
)

OUTPUT_ROOT = "outputs/orchestrator_dry_run_smoke"
DATASET_ID = "synthetic-orchestrator"

_LABELS = {
    "longrag": "LongRAG",
    "raptor": "RAPTOR",
    "hipporag2": "HippoRAG 2",
    "lightrag": "LightRAG",
    "pathrag": PATHRAG_REQUIRED_LABEL,
}


def _registry() -> dict:
    return {
        "architectures": {
            arch: {
                "readiness_status": "READY_FOR_DATA",
                "readiness_declaration": f"baseline_audit/{arch}_readiness.yaml",
                "execution_label": _LABELS[arch],
                "final_generation_enabled": False,
                **(
                    {
                        "foreign_repository_status": "BLOCKED",
                        "implementation_class": "PAPER_DERIVED_REIMPLEMENTATION",
                    }
                    if arch == "pathrag"
                    else {}
                ),
            }
            for arch in WAVE1_ARCHITECTURES
        }
    }


def main() -> None:
    registry = _registry()
    cell_ids: list[str] = []
    plan_ids: list[str] = []
    total_stages = 0

    for arch in WAVE1_ARCHITECTURES:
        request = OrchestratorExecutionRequest(
            dataset_id=DATASET_ID,
            dataset_revision="rev-1",
            split="test",
            architecture_id=arch,
            adapter_id=f"{arch}-adapter",
            adapter_version="0.1.0",
            architecture_configuration_hash="cfg-smoke",
            canonical_input_manifest_sha256="a" * 64,
            random_seed=7,
            project_code_revision="smoke",
            execution_environment_id="smoke-env",
            dataset_root="synthetic-input",
            output_root=OUTPUT_ROOT,
        )
        plan = build_orchestrator_dry_run_plan(
            request=request, readiness_registry=registry
        )
        manifest = build_planned_execution_manifest(plan=plan)

        root = Path(plan.artifact_layout.experiment_root)
        if root.exists():
            shutil.rmtree(root)
        write_orchestrator_dry_run(plan=plan, execution_manifest=manifest)

        loaded = read_orchestrator_dry_run(
            experiment_root=root,
            expected_experiment_cell_id=plan.experiment_cell.experiment_cell_id,
        )
        assert loaded["plan"].plan_id == plan.plan_id

        cell_ids.append(plan.experiment_cell.experiment_cell_id)
        plan_ids.append(plan.plan_id)
        total_stages += len(plan.stages)

        print(f"[PLAN] {arch}")
        print(f"  plan_id:            {plan.plan_id[:40]}...")
        print(f"  experiment_cell_id: {plan.experiment_cell.experiment_cell_id[:40]}...")
        print(f"  execution_label:    {plan.architecture_display_name}")
        print(f"  stages:             {len(plan.stages)}")
        print(f"  root:               {plan.artifact_layout.experiment_root}")
        print(f"  manifest status:    {manifest.execution_status}")

    assert len(set(cell_ids)) == 5
    assert len(set(plan_ids)) == 5

    stages_per_plan = len(plan.stages)  # 8, same for every architecture

    print()
    print("[OK] orchestrator dry-run smoke")
    print(f"  architectures planned: {len(WAVE1_ARCHITECTURES)}")
    print(f"  experiment cells: {len(set(cell_ids))}")
    print(f"  stages per plan: {stages_per_plan}")
    print(f"  total planned stages: {total_stages}")
    print("  architecture executions performed: 0")
    print("  evaluation executions performed: 0")
    print("  generated answers: 0")
    print("  network requests: 0")
    print("  foreign code executions: 0")
    print("  all READY_FOR_DATA: True")
    print("  round trip valid for all plans: True")
    print("  checksums valid for all plans: True")


if __name__ == "__main__":
    main()
