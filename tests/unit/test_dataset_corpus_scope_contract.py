"""Tests for the Step 92 dataset corpus-scope contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "dataset_corpus_scope_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_dataset_corpus_scope_contract.py"

EXPECTED = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_corpus_scope():
    assert _run().returncode == 0


def test_every_dataset_has_one_scope():
    d = _load()["datasets"]
    for name in EXPECTED:
        assert name in d, name
        assert d[name]["expected_scope"]


def test_ruler_query_specific():
    assert _load()["datasets"]["ruler"]["expected_scope"] == "QUERY_SPECIFIC_CORPUS"


def test_cuad_shared_corpus():
    assert _load()["datasets"]["cuad"]["expected_scope"] == "SHARED_DOMAIN_CORPUS"


def test_clerk_global_corpus():
    assert _load()["datasets"]["clerk"]["expected_scope"] == "GLOBAL_DATASET_CORPUS"


def test_triviality_risks_recorded():
    risks = _load()["trivial_document_selection_risks"]
    for name in EXPECTED:
        assert name in risks
    assert risks["ruler"] == "HIGH"