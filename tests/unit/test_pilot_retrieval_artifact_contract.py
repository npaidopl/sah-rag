"""Tests for the Step 86 retrieval artifact contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_retrieval_artifact_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_retrieval_artifact_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_artifact_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_ranked_child_methods_exact():
    methods = _load()["ranked_children"]["retrieval_methods"]
    assert methods == ["dense", "sparse", "hybrid_rrf"]


def test_ranks_start_at_one_and_contiguous():
    rc = _load()["ranked_children"]["constraints"]
    assert rc["rank_starts_at_one"] is True
    assert rc["ranks_contiguous_per_query_and_method"] is True


def test_evaluation_fields_prohibited():
    prohibited = set(_load()["prohibited_fields"])
    for name in ("generated_answer", "relevance", "judgment", "evaluation_metric"):
        assert name in prohibited, name


def test_answer_fields_prohibited():
    prohibited = set(_load()["prohibited_fields"])
    assert "reference_answer" in prohibited
    assert "answer_text" in prohibited