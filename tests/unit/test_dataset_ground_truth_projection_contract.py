"""Tests for the Step 92 ground-truth projection contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "dataset_ground_truth_projection_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_dataset_ground_truth_projection_contract.py"

EXPECTED = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_gt_projection():
    assert _run().returncode == 0


def test_every_dataset_has_primary_unit():
    d = _load()["datasets"]
    for name in EXPECTED:
        assert d[name]["primary_evaluation_unit"]


def test_ground_truth_not_available_to_retrieval():
    r = _load()["global_rules"]
    assert r["ground_truth_available_to_retrieval"] is False
    assert r["ground_truth_available_to_indexing"] is False
    assert r["ground_truth_available_to_querying"] is False
    assert r["ground_truth_available_only_to_evaluator"] is True


def test_cuad_zero_relevance_supported():
    assert _load()["datasets"]["cuad"]["zero_relevance_queries_supported"] is True


def test_duplicate_document_identities_remain_distinct():
    assert (
        _load()["global_rules"]["duplicate_document_identities_remain_distinct"]
        is True
    )