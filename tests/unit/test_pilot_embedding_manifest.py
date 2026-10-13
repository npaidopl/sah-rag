"""Tests for the pilot embedding manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_embedding_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_embedding_manifest.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def test_validator_accepts_manifest():
    r = subprocess.run([sys.executable, str(VALIDATOR)],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_embedding_execution_not_approved():
    ap = _load()["approval"]
    assert ap["embedding_execution_approved"] is False
    assert ap["model_download_approved"] is False


def test_cache_status_verified():
    assert _load()["model"]["cache_status"] == "VERIFIED"
