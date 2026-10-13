"""Tests for the architecture environment matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / "baseline_audit" / "architecture_environment_matrix.yaml"
PORTFOLIO = REPO_ROOT / "baseline_audit" / "architecture_portfolio_dispositions.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_architecture_environment_matrix.py"

EXPECTED = {
    "longrag", "raptor", "hipporag2", "lightrag", "pathrag",
    "disretrieval", "memorag", "kag",
    "metadata_rag", "hikey",
}
READY = {"longrag", "raptor", "hipporag2", "lightrag", "pathrag"}
PILOT_ELIGIBLE_STATUSES = {
    "EXECUTABLE_ON_CURRENT_HOST",
    "EXECUTABLE_ON_CURRENT_HOST_WITH_LIMITS",
}


def _load() -> dict:
    return yaml.safe_load(MATRIX.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_current_matrix():
    r = _run_validator()
    assert r.returncode == 0, r.stderr
    assert "[OK] Architecture environment matrix" in r.stdout


def test_ten_entries_present():
    data = _load()
    assert set(data["architectures"]) == EXPECTED


def test_portfolio_statuses_match_step_75():
    matrix = _load()
    portfolio = yaml.safe_load(PORTFOLIO.read_text(encoding="utf-8"))
    for aid, entry in matrix["architectures"].items():
        assert entry["portfolio_status"] == portfolio["architectures"][aid]["operational_status"]


def test_pilot_eligible_is_subset_of_ready():
    data = _load()
    for aid, entry in data["architectures"].items():
        if entry["pilot_eligible"]:
            assert aid in READY, aid
            assert entry["environment_status"] in PILOT_ELIGIBLE_STATUSES
            assert entry["blockers"] == []


def test_pathrag_foreign_not_executable():
    data = _load()
    prohibited = data["architectures"]["pathrag"]["prohibited_components"]
    assert "foreign PathRAG repository (BLOCKED)" in prohibited


def test_disretrieval_not_executable():
    data = _load()
    assert data["architectures"]["disretrieval"]["environment_status"] == "NOT_EXECUTABLE"


def test_kag_requires_service_environment():
    data = _load()
    assert data["architectures"]["kag"]["environment_status"] == "REQUIRES_SERVICE_ENVIRONMENT"


def test_hikey_requires_multimodal_contract():
    data = _load()
    hikey = data["architectures"]["hikey"]
    assert hikey["environment_status"] == "REQUIRES_MULTIMODAL_CONTRACT"
    assert "MULTIMODAL_CONTRACT_MISSING" in hikey["blockers"]
