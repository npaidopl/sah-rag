"""Tests for the infrastructure pilot manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "baseline_audit" / "infrastructure_pilot_manifest.yaml"
HOST = REPO_ROOT / "baseline_audit" / "execution_host_profile.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_infrastructure_pilot_manifest.py"


def _load() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_manifest():
    r = _run_validator()
    assert r.returncode == 0, r.stderr
    assert "[OK] Infrastructure pilot manifest" in r.stdout


def test_host_profile_id_matches():
    data = _load()
    host = yaml.safe_load(HOST.read_text(encoding="utf-8"))
    assert data["host"]["profile_id"] == host["profile"]["profile_id"]


def test_dataset_materialization_state_consistent():
    # After Step 78 the dataset may be materialized. The two flags
    # must always agree with each other, and materialization must
    # never imply execution approval.
    data = _load()
    ds = data["dataset"]
    assert ds["dataset_materialized"] is ds["canonical_manifest_available"]


def test_experiment_and_plan_ids_remain_null():
    data = _load()
    for aid, entry in data["architectures"].items():
        assert entry["experiment_cell_id"] is None, aid
        assert entry["plan_id"] is None, aid
        assert entry["status"] == "NOT_PLANNED", aid


def test_execution_approvals_are_state_aware():
    # materialization_approved may be True after Step 78.
    # execution_approved becomes True only after the pilot closes
    # (Step 90). publishable_execution_approved and
    # future_execution_approved must always remain False.
    data = _load()
    status = data["pilot"]["status"]
    ap = data["approval"]
    if status == "PLANNED":
        assert ap["execution_approved"] is False
    else:
        assert ap["execution_approved"] is True
    assert ap["publishable_execution_approved"] is False
    if "future_execution_approved" in ap:
        assert ap["future_execution_approved"] is False
    if "real_dataset_execution_approved" in ap:
        assert ap["real_dataset_execution_approved"] is False


def test_planned_counts_are_consistent():
    data = _load()
    c = data["planned_counts"]
    assert c["architecture_count"] == 4
    assert c["experiment_cell_count"] == 4
    assert c["stage_count_per_architecture"] == 8
    assert c["total_stage_count"] == 32
