"""Tests for the Step 88 evaluation manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_evaluation_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_evaluation_manifest.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_manifest():
    assert _run().returncode == 0


def test_status_valid():
    assert _load()["execution"]["status"] in ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def test_forbidden_approvals_false():
    a = _load()["approval"]
    for flag in (
        "judgment_loading_approved",
        "evaluation_execution_approved",
        "metric_computation_approved",
        "generation_execution_approved",
        "publishable_execution_approved",
    ):
        assert a[flag] is False, flag


def test_next_gate():
    ng = _load()["next_gate"]
    assert ng["required_step"] == "bounded_retrieval_evaluation_execution"
    assert ng["approved"] is False