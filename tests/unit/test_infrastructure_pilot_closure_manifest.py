"""Tests for the Step 90 closure manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "infrastructure_pilot_closure_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_infrastructure_pilot_closure_manifest.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_manifest():
    assert _run().returncode == 0


def test_status_is_valid():
    status = _load()["closure"]["status"]
    assert status in ("PLANNED", "CLOSED", "INTERRUPTED")


def test_claims_boundary():
    claims = _load()["claims"]
    assert claims["infrastructure_chain_completed"] is True
    assert claims["synthetic_data_only"] is True
    for flag in (
        "model_quality_validated",
        "publishable_benchmark_completed",
        "real_dataset_execution_completed",
        "generation_validated",
        "answer_quality_validated",
    ):
        assert claims[flag] is False, flag


def test_next_real_dataset_gate_unapproved():
    ng = _load()["next_gate"]
    assert ng["required_step"] == "real_dataset_inventory_and_sizing"
    assert ng["approved"] is False


def test_approvals_frozen():
    a = _load()["approval"]
    assert a["evidence_collection_approved"] is True
    assert a["closure_validation_approved"] is True
    for flag in (
        "pipeline_reexecution_approved",
        "model_loading_approved",
        "qdrant_access_approved",
        "retrieval_execution_approved",
        "evaluation_execution_approved",
        "generation_execution_approved",
    ):
        assert a[flag] is False, flag