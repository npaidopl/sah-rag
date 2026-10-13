"""Tests for the Step 88 judgment mapping contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_judgment_mapping_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_judgment_mapping_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_mapping_contract():
    assert _run().returncode == 0


def test_six_queries():
    assert _load()["source"]["expected_query_count"] == 6


def test_twelve_judgments():
    assert _load()["source"]["expected_judgment_count"] == 12


def test_two_relevant_docs_per_query():
    assert _load()["source"]["expected_judgments_per_query"] == 2


def test_all_judgment_docs_exist():
    constraints = _load()["constraints"]
    assert constraints["every_judgment_document_exists"] is True
    assert constraints["every_judgment_document_is_canonical"] is True


def test_duplicate_content_documents_distinct():
    assert (
        _load()["constraints"]["duplicate_content_documents_remain_distinct"]
        is True
    )