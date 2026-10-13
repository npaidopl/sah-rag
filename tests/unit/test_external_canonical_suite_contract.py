"""Tests for the Step 92 external canonical suite contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "external_canonical_suite_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_external_canonical_suite_contract.py"

EXPECTED = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_contract():
    assert _run().returncode == 0


def test_exact_seven_dataset_ids():
    assert tuple(_load()["datasets"]["expected"]) == EXPECTED


def test_external_root_absolute_and_read_only():
    s = _load()["suite"]
    assert s["local_canonical_root"].startswith("C:/")
    assert _load()["registration"]["read_only"] is True


def test_expected_counts_frozen():
    s = _load()["suite"]
    assert s["expected_document_count"] == 8879
    assert s["expected_example_count"] == 29718
    assert (
        s["expected_collection_fingerprint"]
        == "5f577978b6ab7a4ee3fef29b68db29f95b37261f103bb6cfd5fd7094c70a2aa4"
    )


def test_all_execution_flags_false():
    e = _load()["execution"]
    for flag in (
        "adapter_executed","parser_executed","chunker_executed",
        "tokenizer_loaded","model_loaded","embedding_executed",
        "qdrant_executed","retrieval_executed","evaluation_executed",
        "generation_executed",
    ):
        assert e[flag] is False, flag


def test_exact_canonical_file_set():
    files = _load()["required_canonical_files"]
    assert set(files) == {"documents.jsonl","examples.jsonl","validation.json","manifest.json"}