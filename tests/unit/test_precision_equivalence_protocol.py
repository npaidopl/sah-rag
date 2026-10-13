"""Tests for the Step 93 precision equivalence protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "precision_equivalence_protocol.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_precision_equivalence_protocol.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_protocol():
    assert _run().returncode == 0


def test_three_stages():
    stages = [s["id"] for s in _load()["stages"]]
    assert "FP32_REFERENCE" in stages
    assert "FP16_CANDIDATE" in stages
    assert "RETRIEVAL_EQUIVALENCE" in stages


def test_thresholds_frozen_before_experiment():
    t = _load()["acceptance_thresholds"]
    assert t["median_same_record_cosine_minimum"] == 0.999
    assert t["mean_top_10_overlap_minimum"] == 0.99
    assert t["absolute_recall_at_10_delta_maximum"] == 0.005


def test_decision_rule():
    dr = _load()["decision_rule"]
    assert dr["if_pass"]["production_precision"] == "FP16"
    assert dr["if_fail"]["production_precision"] == "FP32"