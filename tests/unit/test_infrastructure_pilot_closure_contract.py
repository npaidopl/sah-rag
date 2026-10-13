"""Tests for the Step 90 closure contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "infrastructure_pilot_closure_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_infrastructure_pilot_closure_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_closure_contract():
    assert _run().returncode == 0


def test_synthetic_only_scope():
    s = _load()["scope"]
    assert s["synthetic_data_only"] is True
    assert s["real_dataset_used"] is False


def test_full_execution_chain_frozen():
    chain = _load()["required_execution_chain"]
    assert len(chain) == 13
    assert "canonical_envelope" in chain
    assert "common_retrieval_evaluation" in chain


def test_publishable_claims_prohibited():
    s = _load()["scope"]
    assert s["publishable_execution"] is False


def test_repeat_execution_not_required():
    assert _load()["closure"]["repeat_execution_required"] is False


def test_real_dataset_not_authorized():
    s = _load()["safety"]
    assert s["real_dataset_execution"] is False
    assert s["publishable_metrics_produced"] is False