"""Tests for the Step 94 architecture readiness gap manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "architecture_readiness_gap_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_architecture_readiness_gap_manifest.py"


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
    assert _load()["execution"]["status"] in ("PLANNED","COMPLETE","INTERRUPTED")


def test_execution_flags_false():
    a = _load()["approval"]
    for f in ("architecture_execution_approved","model_loading_approved",
              "llm_execution_approved","storage_creation_approved",
              "retrieval_execution_approved","evaluation_execution_approved"):
        assert a[f] is False, f


def test_next_gate_correct():
    ng = _load()["next_gate"]
    assert ng["required_step"] == "first_architecture_readiness_completion"
    assert ng["approved"] is False