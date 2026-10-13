"""Tests for the pilot Colab transfer protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_colab_transfer_protocol.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_colab_transfer_protocol.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def test_validator_accepts_protocol():
    r = subprocess.run([sys.executable, str(VALIDATOR)],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_prohibited_list_contains_secrets():
    prohibited = set(_load()["transfer"]["prohibited"])
    assert "api keys" in prohibited
    assert "access tokens" in prohibited
    assert "model weights" in prohibited
