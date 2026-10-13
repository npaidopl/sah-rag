"""Tests for the pilot Qdrant collection plan."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_qdrant_collection_plan.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_qdrant_collection_plan.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def test_validator_accepts_plan():
    r = subprocess.run([sys.executable, str(VALIDATOR)],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dense_and_sparse_names_frozen():
    v = _load()["vectors"]
    assert v["dense"]["name"] == "dense"
    assert v["sparse"]["name"] == "sparse"
