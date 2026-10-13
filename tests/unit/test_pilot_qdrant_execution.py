"""Tests for the Step 85 execution manifest state and helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_manifest.yaml"
MANIFEST_VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_qdrant_execution_manifest.py"
RUNNER = REPO_ROOT / "scripts" / "run_pilot_qdrant_ingestion.py"


def _load_manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _load_runner():
    spec = importlib.util.spec_from_file_location("_pilot_qdrant_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_manifest_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MANIFEST_VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_manifest_validator_accepts_current_state():
    r = _run_manifest_validator()
    assert r.returncode == 0, r.stderr


def test_manifest_status_is_valid():
    status = _load_manifest()["execution"]["status"]
    assert status in ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def test_manifest_approvals_frozen():
    approval = _load_manifest()["approval"]
    assert approval["collection_creation_approved"] is True
    assert approval["vector_ingestion_approved"] is True
    for flag in (
        "retrieval_execution_approved",
        "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        assert approval[flag] is False, flag


def test_planned_point_ids_helper_is_deterministic():
    runner = _load_runner()
    records = [{"chunk_id": "chunk-A"}, {"chunk_id": "chunk-A"}, {"chunk_id": "chunk-B"}]
    a = runner.planned_point_ids(records)
    b = runner.planned_point_ids(records)
    assert a == b
    assert len(a) == 2


def test_planned_point_ids_produces_twelve_unique_for_twelve_records():
    runner = _load_runner()
    records = [{"chunk_id": f"chunk-{i:05d}"} for i in range(12)]
    plan = runner.planned_point_ids(records)
    assert len(plan) == 12
    assert len(set(plan.values())) == 12


def test_storage_path_safety_rejects_outside_paths(tmp_path: Path):
    runner = _load_runner()
    pilot_root = tmp_path / "outputs" / "infrastructure_pilot"
    pilot_root.mkdir(parents=True)
    inside = pilot_root / "infrastructure-pilot-synthetic-v1" / "rag_system" / "qdrant"
    outside = tmp_path / "elsewhere" / "qdrant"

    # Inside path must pass
    runner.assert_storage_path_safe(inside, pilot_root)

    # Outside path must fail (SystemExit)
    raised = False
    try:
        runner.assert_storage_path_safe(outside, pilot_root)
    except SystemExit:
        raised = True
    assert raised, "assert_storage_path_safe must reject paths outside pilot_root"


def test_runner_does_not_reference_retrieval_functions():
    source = RUNNER.read_text(encoding="utf-8")
    for forbidden in (
        "retrieve_dense(",
        "retrieve_sparse(",
        "retrieve_hybrid_rrf(",
        "retrieve_hybrid_parents(",
    ):
        assert forbidden not in source, forbidden