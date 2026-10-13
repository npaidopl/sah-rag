"""Tests for the Step 92 external canonical suite manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "external_canonical_suite_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_external_canonical_suite_manifest.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_manifest():
    assert _run().returncode == 0


def test_planned_and_succeeded_states_valid():
    status = _load()["execution"]["status"]
    assert status in ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def test_all_execution_flags_remain_false():
    r = _load()["results"]
    for flag in (
        "adapter_executed","parser_executed","chunker_executed",
        "tokenizer_loaded","model_loaded","embedding_executed",
        "retrieval_executed","evaluation_executed","generation_executed",
    ):
        assert r[flag] is False, flag