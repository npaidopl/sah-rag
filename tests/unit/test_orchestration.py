"""Tests for the common orchestrator skeleton (planner)."""

from __future__ import annotations

import pytest

from rag_system.orchestration import (
    PATHRAG_REQUIRED_LABEL,
    STAGE_SEQUENCE,
    WAVE1_ARCHITECTURES,
    OrchestratorExecutionRequest,
    build_orchestrator_dry_run_plan,
    build_planned_execution_manifest,
    validate_orchestrator_dry_run_plan,
)


def _registry() -> dict:
    return {
        "architectures": {
            "longrag": {
                "readiness_status": "READY_FOR_DATA",
                "readiness_declaration": "baseline_audit/longrag_readiness.yaml",
                "execution_label": "LongRAG",
                "final_generation_enabled": False,
            },
            "raptor": {
                "readiness_status": "READY_FOR_DATA",
                "readiness_declaration": "baseline_audit/raptor_readiness.yaml",
                "execution_label": "RAPTOR",
                "final_generation_enabled": False,
            },
            "hipporag2": {
                "readiness_status": "READY_FOR_DATA",
                "readiness_declaration": "baseline_audit/hipporag2_readiness.yaml",
                "execution_label": "HippoRAG 2",
                "final_generation_enabled": False,
            },
            "lightrag": {
                "readiness_status": "READY_FOR_DATA",
                "readiness_declaration": "baseline_audit/lightrag_readiness.yaml",
                "execution_label": "LightRAG",
                "final_generation_enabled": False,
            },
            "pathrag": {
                "readiness_status": "READY_FOR_DATA",
                "readiness_declaration": "baseline_audit/pathrag_readiness.yaml",
                "execution_label": PATHRAG_REQUIRED_LABEL,
                "final_generation_enabled": False,
                "foreign_repository_status": "BLOCKED",
                "implementation_class": "PAPER_DERIVED_REIMPLEMENTATION",
            },
        }
    }


def _request(**overrides) -> OrchestratorExecutionRequest:
    kwargs = dict(
        dataset_id="clerk",
        dataset_revision="rev-1",
        split="test",
        architecture_id="longrag",
        adapter_id="longrag-adapter",
        adapter_version="0.1.0",
        architecture_configuration_hash="cfg",
        canonical_input_manifest_sha256="a" * 64,
        random_seed=42,
        project_code_revision="code-rev",
        execution_environment_id="local-windows-cpu",
        dataset_root="data/clerk",
        output_root="outputs/runs",
    )
    kwargs.update(overrides)
    return OrchestratorExecutionRequest(**kwargs)


def _plan(**overrides):
    return build_orchestrator_dry_run_plan(
        request=_request(**overrides), readiness_registry=_registry()
    )


def test_deterministic_plan_id():
    assert _plan().plan_id == _plan().plan_id


def test_deterministic_experiment_cell_id():
    assert (
        _plan().experiment_cell.experiment_cell_id
        == _plan().experiment_cell.experiment_cell_id
    )


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError):
        OrchestratorExecutionRequest(
            dataset_id="x",
            dataset_revision="y",
            split="test",
            architecture_id="not-wave1",
            adapter_id="a",
            adapter_version="1",
            architecture_configuration_hash="h",
            canonical_input_manifest_sha256="a" * 64,
            random_seed=0,
            project_code_revision="r",
            execution_environment_id="e",
            dataset_root="d",
            output_root="o",
        )


def test_non_ready_architecture_rejected():
    reg = _registry()
    reg["architectures"]["longrag"]["readiness_status"] = "BLOCKED"
    with pytest.raises(ValueError, match="READY_FOR_DATA"):
        build_orchestrator_dry_run_plan(
            request=_request(), readiness_registry=reg
        )


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        _request(random_seed=-1)


def test_artifact_paths_remain_under_root():
    plan = _plan()
    root = plan.artifact_layout.experiment_root.rstrip("/")
    for path in (
        plan.artifact_layout.input_directory,
        plan.artifact_layout.architecture_directory,
        plan.artifact_layout.retrieval_directory,
        plan.artifact_layout.evaluation_directory,
        plan.artifact_layout.logs_directory,
        plan.artifact_layout.checkpoints_directory,
        plan.artifact_layout.dry_run_plan_path,
        plan.artifact_layout.execution_manifest_path,
        plan.artifact_layout.plan_manifest_path,
        plan.artifact_layout.checksums_path,
    ):
        assert path.startswith(root + "/"), path


def test_exactly_eight_stages():
    plan = _plan()
    assert len(plan.stages) == len(STAGE_SEQUENCE) == 8


def test_stage_orders_contiguous():
    plan = _plan()
    assert [s.stage_order for s in plan.stages] == list(range(1, 9))


def test_dependencies_valid():
    plan = _plan()
    ids = {s.stage_id for s in plan.stages}
    for s in plan.stages:
        for dep in s.depends_on:
            assert dep in ids


def test_all_stage_execution_flags_false():
    plan = _plan()
    assert all(s.execution_enabled is False for s in plan.stages)


def test_pathrag_label_exact():
    plan = _plan(architecture_id="pathrag")
    assert plan.architecture_display_name == PATHRAG_REQUIRED_LABEL
    assert plan.architecture_classification == "ARCHITECTURAL_COMPARATOR"
    assert plan.implementation_class == "PAPER_DERIVED_REIMPLEMENTATION"


def test_pathrag_foreign_request_rejected():
    with pytest.raises(ValueError, match="foreign implementation"):
        _request(
            architecture_id="pathrag",
            metadata={"foreign_implementation": True},
        )


def test_planned_manifest_has_zero_resource_usage():
    plan = _plan()
    manifest = build_planned_execution_manifest(plan=plan)
    assert manifest.execution_status == "PLANNED"
    assert manifest.completed_at_utc is None
    assert manifest.failure is None
    ru = manifest.resource_usage
    assert ru.total_execution_seconds == 0.0
    assert ru.document_count == 0
    assert ru.query_count == 0
    assert manifest.final_generation_enabled is False
    assert manifest.answer_evaluation_enabled is False
    validate_orchestrator_dry_run_plan(plan)


def test_all_wave1_architectures_plan():
    for arch in WAVE1_ARCHITECTURES:
        plan = _plan(architecture_id=arch)
        assert plan.request.architecture_id == arch
        assert len(plan.stages) == 8
