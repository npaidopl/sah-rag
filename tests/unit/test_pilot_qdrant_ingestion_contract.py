"""Tests for the pilot Qdrant ingestion contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_qdrant_ingestion_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_qdrant_ingestion_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_backend_is_qdrant_local_persistence():
    data = _load()
    assert data["backend"]["type"] == "Qdrant"
    assert data["backend"]["mode"] == "local_persistence"
    assert data["backend"]["server_required"] is False
    assert data["backend"]["docker_required"] is False
    assert data["backend"]["network_required"] is False


def test_expected_point_count_is_twelve():
    assert _load()["collection"]["expected_point_count"] == 12


def test_dense_vector_contract_frozen():
    dense = _load()["vectors"]["dense"]
    assert dense["name"] == "dense"
    assert dense["dimension"] == 1024
    assert dense["distance"] == "COSINE"


def test_sparse_vector_contract_frozen():
    sparse = _load()["vectors"]["sparse"]
    assert sparse["name"] == "sparse"
    assert sparse["indices_sorted"] is True
    assert sparse["indices_unique"] is True
    assert sparse["values_non_negative"] is True


def test_all_execution_flags_false():
    execution = _load()["execution"]
    for flag in (
        "collection_creation_enabled",
        "vector_ingestion_enabled",
        "retrieval_enabled",
        "evaluation_enabled",
        "generation_enabled",
    ):
        assert execution[flag] is False, flag