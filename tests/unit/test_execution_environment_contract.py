"""Tests for the execution environment contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "baseline_audit" / "execution_environment_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_execution_environment_contract.py"


def _load() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_current_contract():
    r = _run_validator()
    assert r.returncode == 0, r.stderr
    assert "[OK] Execution environment contract" in r.stdout


def test_status_vocabulary_has_seven_values():
    data = _load()
    assert len(data["execution_status_vocabulary"]["allowed"]) == 7


def test_blocker_vocabulary_is_frozen():
    data = _load()
    expected = {
        "MISSING_GPU", "INSUFFICIENT_VRAM", "MISSING_SERVICE",
        "MISSING_MODEL_LICENSE", "MODEL_NOT_APPROVED", "NETWORK_POLICY_BLOCKED",
        "HOSTED_COST_NOT_APPROVED", "MULTIMODAL_CONTRACT_MISSING",
        "DATASET_SOURCE_FORMAT_UNVERIFIED", "FOREIGN_EXECUTION_BLOCKED",
        "IMPLEMENTATION_NOT_READY",
    }
    assert set(data["blocker_vocabulary"]["allowed"]) == expected


def test_publishable_execution_remains_false():
    data = _load()
    assert data["pilot_policy"]["publishable_execution_allowed"] is False


def test_pilot_only_accepts_ready_architectures():
    data = _load()
    assert set(data["pilot_policy"]["eligible_portfolio_status"]) == {"READY_FOR_DATA"}
