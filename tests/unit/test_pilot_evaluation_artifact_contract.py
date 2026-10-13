"""Tests for the Step 88 artifact contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_evaluation_artifact_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_evaluation_artifact_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_artifact_contract():
    assert _run().returncode == 0


def test_required_files_exact():
    files = set(_load()["outputs"]["planned_files"])
    assert files == {
        "retrieval_cases.jsonl",
        "query_metrics.jsonl",
        "aggregate_metrics.json",
        "evaluation.manifest.json",
        "checksums.json",
    }


def test_generated_answers_prohibited():
    assert "generated_answers" in _load()["prohibited_outputs"]


def test_ranking_and_comparison_prohibited():
    p = _load()["prohibited_outputs"]
    assert "architecture_ranking" in p
    assert "publishable_comparison" in p


def test_deterministic_ordering_frozen():
    o = _load()["ordering"]
    assert o["retrieval_cases"] == ["query_id"]
    assert o["query_metrics"] == ["query_id", "metric_name", "cutoff"]
    assert o["aggregate_metrics"] == ["metric_name", "cutoff"]