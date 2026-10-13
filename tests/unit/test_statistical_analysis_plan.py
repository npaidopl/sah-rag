"""Tests for the Step 93 statistical analysis plan."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "statistical_analysis_plan.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_statistical_analysis_plan.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_plan():
    assert _run().returncode == 0


def test_macro_aggregation_not_micro():
    a = _load()["aggregation"]
    assert a["per_dataset"]["method"] == "macro_average_over_queries"
    assert a["cross_dataset"]["method"] == "macro_average_over_datasets"
    assert a["global_micro_average"]["primary"] is False


def test_paired_bootstrap_uncertainty():
    u = _load()["pairwise_comparison"]["uncertainty"]
    assert u["method"] == "paired_bootstrap"
    assert u["resamples"] == 10000


def test_holm_correction():
    assert _load()["pairwise_comparison"]["multiple_testing"]["correction"] == "HOLM"


def test_context_bins_five():
    cl = _load()["context_length_analysis"]
    assert len(cl["candidate_bins"]) == 5


def test_cuad_zero_subset_separate():
    zr = _load()["zero_relevance_reporting"]["cuad_zero_evidence_subset"]
    assert zr["separate_report"] is True
    assert zr["never_mixed_with_recall_metrics"] is True