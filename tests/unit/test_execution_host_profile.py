"""Tests for the execution host profile."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / "baseline_audit" / "execution_host_profile.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_execution_host_profile.py"


def _load() -> dict:
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_current_profile():
    r = _run_validator()
    assert r.returncode == 0, r.stderr
    assert "[OK] Execution host profile" in r.stdout


def test_profile_id_prefix():
    data = _load()
    assert data["profile"]["profile_id"].startswith("execution-host-")


def test_hardware_counts_non_negative():
    p = _load()["profile"]
    assert p["physical_cpu_count"] >= 0
    assert p["logical_cpu_count"] >= 0
    assert p["system_ram_bytes"] >= 0


def test_gpu_false_implies_empty_devices():
    p = _load()["profile"]
    if p["gpu_available"] is False:
        assert p["gpu_count"] == 0
        assert p["gpu_devices"] == []


def test_cost_is_non_negative():
    data = _load()
    assert data["policy"]["maximum_approved_cost"]["amount"] >= 0


def test_secrets_are_not_available():
    data = _load()
    assert data["policy"]["secrets_available"] is False


def test_no_baseline_model_or_dataset_executed():
    c = _load()["collection"]
    assert c["baseline_execution_performed"] is False
    assert c["model_loaded"] is False
    assert c["dataset_loaded"] is False
    assert c["network_probe_performed"] is False
    assert c["sensitive_identifiers_collected"] is False
