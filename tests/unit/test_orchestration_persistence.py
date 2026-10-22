"""Tests for orchestrator dry-run persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestration import (
    PATHRAG_REQUIRED_LABEL,
    OrchestratorExecutionRequest,
    build_orchestrator_dry_run_plan,
    build_planned_execution_manifest,
)
from orchestration.persistence import (
    read_orchestrator_dry_run,
    write_orchestrator_dry_run,
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


def _request(tmp_path: Path, *, root: str | None = None) -> OrchestratorExecutionRequest:
    return OrchestratorExecutionRequest(
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
        dataset_root=str(tmp_path / "data"),
        output_root=root or str(tmp_path / "runs"),
    )


def _plan(tmp_path: Path, *, root: str | None = None):
    return build_orchestrator_dry_run_plan(
        request=_request(tmp_path, root=root), readiness_registry=_registry()
    )


def test_round_trip_succeeds(tmp_path: Path):
    plan = _plan(tmp_path)
    manifest = build_planned_execution_manifest(plan=plan)
    result = write_orchestrator_dry_run(plan=plan, execution_manifest=manifest)
    loaded = read_orchestrator_dry_run(
        experiment_root=Path(plan.artifact_layout.experiment_root),
        expected_experiment_cell_id=plan.experiment_cell.experiment_cell_id,
    )
    assert loaded["plan"].plan_id == plan.plan_id
    assert set(result["checksums_payload"]) == {
        "dry-run.plan.json",
        "execution.manifest.json",
        "dry-run.manifest.json",
    }


def test_existing_planned_run_rejected_without_overwrite(tmp_path: Path):
    plan = _plan(tmp_path)
    manifest = build_planned_execution_manifest(plan=plan)
    write_orchestrator_dry_run(plan=plan, execution_manifest=manifest)
    with pytest.raises(FileExistsError):
        write_orchestrator_dry_run(plan=plan, execution_manifest=manifest)


def test_planned_run_overwritten_with_flag(tmp_path: Path):
    plan = _plan(tmp_path)
    manifest = build_planned_execution_manifest(plan=plan)
    write_orchestrator_dry_run(plan=plan, execution_manifest=manifest)
    write_orchestrator_dry_run(
        plan=plan, execution_manifest=manifest, overwrite_planned=True
    )


def test_existing_running_run_rejected(tmp_path: Path):
    plan = _plan(tmp_path)
    manifest = build_planned_execution_manifest(plan=plan)
    running = manifest.model_copy(update={"execution_status": "RUNNING"})
    root = Path(plan.artifact_layout.experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "execution.manifest.json").write_text(
        running.model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="RUNNING"):
        write_orchestrator_dry_run(
            plan=plan, execution_manifest=manifest, overwrite_planned=True
        )


def test_checksum_mismatch_rejected(tmp_path: Path):
    plan = _plan(tmp_path)
    manifest = build_planned_execution_manifest(plan=plan)
    write_orchestrator_dry_run(plan=plan, execution_manifest=manifest)
    p = Path(plan.artifact_layout.dry_run_plan_path)
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        read_orchestrator_dry_run(
            experiment_root=Path(plan.artifact_layout.experiment_root)
        )


def test_manifest_bytes_identical_across_different_output_roots(tmp_path: Path):
    """The execution manifest does not embed output_root, so two plans
    built from the same request but targeting different roots must
    produce byte-identical manifests."""
    p1 = _plan(tmp_path, root=str(tmp_path / "a"))
    p2 = _plan(tmp_path, root=str(tmp_path / "b"))
    m1 = build_planned_execution_manifest(plan=p1)
    m2 = build_planned_execution_manifest(plan=p2)
    write_orchestrator_dry_run(plan=p1, execution_manifest=m1)
    write_orchestrator_dry_run(plan=p2, execution_manifest=m2)
    a = (Path(p1.artifact_layout.experiment_root) / "execution.manifest.json").read_bytes()
    b = (Path(p2.artifact_layout.experiment_root) / "execution.manifest.json").read_bytes()
    assert a == b


def test_plan_and_manifest_bytes_identical_for_same_request_and_root(tmp_path: Path):
    """For identical request and identical output root, two plans and
    their manifests must be byte-identical. checksums.json is derived
    from those two files and is therefore also byte-identical."""
    root = str(tmp_path / "runs")
    p1 = _plan(tmp_path, root=root)
    p2 = _plan(tmp_path, root=root)
    m1 = build_planned_execution_manifest(plan=p1)
    m2 = build_planned_execution_manifest(plan=p2)
    assert p1.model_dump_json() == p2.model_dump_json()
    assert m1.model_dump_json() == m2.model_dump_json()
    write_orchestrator_dry_run(plan=p1, execution_manifest=m1)
    # second write to same root requires the explicit overwrite flag
    write_orchestrator_dry_run(
        plan=p2, execution_manifest=m2, overwrite_planned=True
    )


def test_pathrag_plan_persists_and_reloads(tmp_path: Path):
    plan = build_orchestrator_dry_run_plan(
        request=OrchestratorExecutionRequest(
            dataset_id="clerk",
            dataset_revision="rev-1",
            split="test",
            architecture_id="pathrag",
            adapter_id="pathrag-adapter",
            adapter_version="0.1.0",
            architecture_configuration_hash="cfg",
            canonical_input_manifest_sha256="a" * 64,
            random_seed=42,
            project_code_revision="code-rev",
            execution_environment_id="local-windows-cpu",
            dataset_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "runs"),
        ),
        readiness_registry=_registry(),
    )
    manifest = build_planned_execution_manifest(plan=plan)
    write_orchestrator_dry_run(plan=plan, execution_manifest=manifest)
    loaded = read_orchestrator_dry_run(
        experiment_root=Path(plan.artifact_layout.experiment_root)
    )
    assert loaded["plan"].architecture_display_name == PATHRAG_REQUIRED_LABEL
