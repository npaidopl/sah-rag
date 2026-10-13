"""Tests for the Step 86 retrieval manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_retrieval_manifest.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_retrieval_manifest.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_manifest():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_status_is_valid():
    status = _load()["execution"]["status"]
    assert status in ("PLANNED", "SUCCEEDED", "INTERRUPTED")


def test_forbidden_approvals_false():
    approval = _load()["approval"]
    for flag in (
        "query_embedding_approved",
        "dense_retrieval_approved",
        "sparse_retrieval_approved",
        "hybrid_retrieval_approved",
        "parent_aggregation_approved",
        "document_expansion_approved",
        "evaluation_approved",
        "generation_approved",
    ):
        assert approval[flag] is False, flag


def test_next_gate_requires_bounded_execution():
    next_gate = _load()["next_gate"]
    assert next_gate["required_step"] == "bounded_hybrid_retrieval_execution"
    assert next_gate["approved"] is False