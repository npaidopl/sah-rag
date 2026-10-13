"""Tests for the Step 86 parent resolution contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_parent_resolution_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_parent_resolution_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_parent_resolution_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_parent_aggregation_explicit():
    pr = _load()["parent_resolution"]
    assert pr["aggregation_method"] == "MAX_SCORE_PLUS_WEIGHTED_ADDITIONAL_CHILD_SCORE"
    assert pr["additional_child_weight"] == 0.05


def test_parent_tie_break_explicit():
    ptb = _load()["parent_tie_break"]
    assert ptb["policy"] == "EXPLICIT_DETERMINISTIC"
    assert ptb["key_ordering"] == [
        "parent_score DESC",
        "best_chunk_rank ASC",
        "parent_id ASC",
    ]


def test_document_aggregation_explicit():
    dr = _load()["document_resolution"]
    assert dr["method"] == "MAX_PARENT_SCORE_PER_DOCUMENT"


def test_duplicate_documents_preserved():
    dr = _load()["document_resolution"]
    assert "Duplicate-content documents are not deduplicated" in " ".join(
        dr["requirements"]
    )