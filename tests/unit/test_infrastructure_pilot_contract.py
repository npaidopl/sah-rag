"""Tests for the infrastructure pilot contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "baseline_audit" / "infrastructure_pilot_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_infrastructure_pilot_contract.py"

INCLUDED = {"longrag", "raptor", "lightrag", "pathrag"}
EXCLUDED = {"hipporag2", "disretrieval", "memorag", "kag", "metadata_rag", "hikey"}


def _load() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def _run_validator() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_contract():
    r = _run_validator()
    assert r.returncode == 0, r.stderr
    assert "[OK] Infrastructure-only pilot contract" in r.stdout


def test_included_architecture_set_is_exact():
    data = _load()
    assert set(data["scope"]["architectures"]) == INCLUDED


def test_excluded_architecture_set_is_exact():
    data = _load()
    assert set(data["scope"]["excluded_architectures"]) == EXCLUDED


def test_dataset_counts_are_frozen():
    data = _load()
    c = data["dataset"]["expected_counts"]
    assert c["document_count"] == 12
    assert c["chunk_count"] == 24
    assert c["query_count"] == 6
    assert c["judgment_count"] == 12
    assert c["duplicate_content_pair_count"] == 6


def test_execution_and_evaluation_disabled():
    data = _load()
    assert data["execution"]["pilot_execution_enabled"] is False
    assert data["execution"]["baseline_execution_enabled"] is False
    assert data["evaluation"]["execution_enabled"] is False
    assert data["retrieval_boundary"]["final_generation_enabled"] is False


def test_pathrag_foreign_execution_is_false():
    data = _load()
    pathrag = data["architecture_constraints"]["pathrag"]
    assert pathrag["foreign_repository_execution_allowed"] is False


def test_publishable_result_is_false():
    data = _load()
    assert data["scope"]["publishable_result"] is False
    assert data["scope"]["included_in_final_comparison"] is False
    assert data["scope"]["retrieval_effectiveness_claimed"] is False
