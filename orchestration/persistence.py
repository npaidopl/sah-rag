"""Deterministic persistence for orchestrator dry-run plans.

Uses the frozen planner API in ``orchestration`` (which is
re-exported from ``orchestration.planner``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sah_rag.artifacts import sha256_file
from orchestration.contracts import ArchitectureExecutionManifest
from orchestration import (
    OrchestratorDryRunPlan,
    OrchestratorPlanManifest,
    build_orchestrator_plan_manifest,
)
from orchestration.planner import _sha256_json


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_orchestrator_dry_run(
    *,
    plan: OrchestratorDryRunPlan,
    execution_manifest: ArchitectureExecutionManifest,
    overwrite_planned: bool = False,
) -> dict[str, Any]:
    """Write the four dry-run artifacts and their directories."""
    layout = plan.artifact_layout
    root = Path(layout.experiment_root)
    plan_path = Path(layout.dry_run_plan_path)
    manifest_path = Path(layout.execution_manifest_path)
    plan_manifest_path = Path(layout.plan_manifest_path)
    checksums_path = Path(layout.checksums_path)

    if execution_manifest.execution_status != "PLANNED":
        raise ValueError("dry-run manifest must have execution_status PLANNED")

    if manifest_path.is_file():
        existing = ArchitectureExecutionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing.execution_status in ("SUCCEEDED", "RUNNING", "INTERRUPTED"):
            raise FileExistsError(
                f"run is {existing.execution_status}; refusing to overwrite"
            )
        if existing.execution_status == "PLANNED" and not overwrite_planned:
            raise FileExistsError(
                "planned run exists; pass overwrite_planned=True to replace"
            )

    root.mkdir(parents=True, exist_ok=True)
    for d in (
        layout.input_directory,
        layout.architecture_directory,
        layout.retrieval_directory,
        layout.evaluation_directory,
        layout.logs_directory,
        layout.checkpoints_directory,
    ):
        Path(d).mkdir(parents=True, exist_ok=True)

    _atomic_write_json(plan_path, plan.model_dump(mode="json"))
    _atomic_write_json(manifest_path, execution_manifest.model_dump(mode="json"))

    plan_manifest = build_orchestrator_plan_manifest(plan=plan)
    _atomic_write_json(plan_manifest_path, plan_manifest.model_dump(mode="json"))

    checksums = {
        "dry-run.plan.json": sha256_file(plan_path),
        "execution.manifest.json": sha256_file(manifest_path),
        "dry-run.manifest.json": sha256_file(plan_manifest_path),
    }
    _atomic_write_json(checksums_path, checksums)

    return {
        "experiment_root": str(root),
        "dry_run_plan": str(plan_path),
        "execution_manifest": str(manifest_path),
        "plan_manifest": str(plan_manifest_path),
        "checksums": str(checksums_path),
        "checksums_payload": checksums,
    }


def read_orchestrator_dry_run(
    *,
    experiment_root: Path,
    expected_experiment_cell_id: str | None = None,
) -> dict[str, Any]:
    root = Path(experiment_root)
    plan_path = root / "dry-run.plan.json"
    manifest_path = root / "execution.manifest.json"
    plan_manifest_path = root / "dry-run.manifest.json"
    checksums_path = root / "checksums.json"

    for label, p in (
        ("dry-run plan", plan_path),
        ("execution manifest", manifest_path),
        ("plan manifest", plan_manifest_path),
        ("checksums", checksums_path),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    plan = OrchestratorDryRunPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    execution_manifest = ArchitectureExecutionManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    plan_manifest = OrchestratorPlanManifest.model_validate_json(
        plan_manifest_path.read_text(encoding="utf-8")
    )
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))

    # file-level integrity from checksums.json
    for name, expected in checksums.items():
        actual = sha256_file(root / name)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {name}")

    # content-level integrity from the plan manifest
    recomputed_plan_sha = _sha256_json(plan.model_dump(mode="json"))
    if recomputed_plan_sha != plan_manifest.plan_sha256:
        raise RuntimeError("plan manifest plan_sha256 mismatch")

    recomputed_request_sha = _sha256_json(plan.request.model_dump(mode="json"))
    if recomputed_request_sha != plan_manifest.request_sha256:
        raise RuntimeError("plan manifest request_sha256 mismatch")

    if (
        expected_experiment_cell_id is not None
        and plan.experiment_cell.experiment_cell_id
        != expected_experiment_cell_id
    ):
        raise RuntimeError("experiment_cell_id mismatch")

    return {
        "plan": plan,
        "execution_manifest": execution_manifest,
        "plan_manifest": plan_manifest,
        "checksums": checksums,
    }
