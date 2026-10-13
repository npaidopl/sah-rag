"""Tests for the Step 85 Qdrant execution contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_qdrant_execution_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_qdrant_execution_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_execution_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_collection_creation_and_ingestion_enabled():
    execution = _load()["execution"]
    assert execution["collection_creation_enabled"] is True
    assert execution["vector_ingestion_enabled"] is True


def test_downstream_stages_disabled():
    execution = _load()["execution"]
    for flag in (
        "retrieval_enabled",
        "evaluation_enabled",
        "generation_enabled",
        "network_access_enabled",
        "server_mode_enabled",
        "docker_execution_enabled",
    ):
        assert execution[flag] is False, flag


def test_expected_point_count_is_twelve():
    assert _load()["collection"]["expected_point_count"] == 12


def test_physical_collection_name_exact():
    assert (
        _load()["collection"]["physical_name"]
        == "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0"
    )


def test_publishability_flags_disabled():
    pub = _load()["publishability"]
    assert pub["publishable_result"] is False
    assert pub["included_in_final_comparison"] is False