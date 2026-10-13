"""Tests for the Step 94 common acceptance contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "architecture_acceptance_test_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_architecture_acceptance_test_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_contract():
    assert _run().returncode == 0


def test_frozen_fixture():
    f = _load()["fixture"]
    assert f["source"] == "infrastructure-pilot-synthetic-v1"
    assert f["documents"] == 12
    assert f["queries"] == 6
    assert f["relevant_documents_per_query"] == 2


def test_persistence_and_reopen_required():
    a = _load()["acceptance"]
    assert a["persistent_state_created"] is True
    assert a["persistent_state_reopened"] is True


def test_retrieval_required_for_all_queries():
    a = _load()["acceptance"]
    assert a["query_count"] == 6
    assert a["retrieval_completed_for_every_query"] is True
    assert a["ranked_results_nonempty"] is True


def test_canonical_identity_required():
    a = _load()["acceptance"]
    assert a["canonical_ids_recovered"] is True
    assert a["duplicate_document_identities_preserved"] is True


def test_generation_prohibited():
    assert _load()["generation"]["enabled"] is False
    for p in ("generated_answers","LLM_judge","test_ground_truth_during_retrieval"):
        assert p in _load()["prohibited"], p