"""Tests for the pilot embedding benchmark protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_benchmark_protocol.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_embedding_benchmark_protocol.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def test_validator_accepts_protocol():
    r = subprocess.run([sys.executable, str(VALIDATOR)],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_batch_sizes_are_1_2_4():
    assert _load()["runs"]["batch_sizes"] == [1, 2, 4]


def test_three_measured_runs_per_batch():
    assert _load()["runs"]["measured_run_count_per_batch_size"] == 3
