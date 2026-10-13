"""Tests for the Step 92 external canonical suite registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "external_canonical_suite_registry.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_external_canonical_suite_registry.py"

EXPECTED = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_registry():
    assert _run().returncode == 0


def test_seven_records():
    d = _load()["datasets"]
    assert set(d.keys()) == set(EXPECTED)


def test_total_documents_and_examples():
    s = _load()["suite"]
    assert s["total_documents"] == 8879
    assert s["total_examples"] == 29718


def test_file_checksums_present():
    d = _load()["datasets"]
    for name, entry in d.items():
        for fname in ("documents.jsonl","examples.jsonl","validation.json","manifest.json"):
            sha = entry["files"][fname]["sha256"]
            assert len(sha) == 64, f"{name}/{fname}"


def test_document_ids_unique_per_dataset():
    d = _load()["datasets"]
    for name, entry in d.items():
        assert entry["document_count_observed"] == entry["document_unique_ids"], name
        assert entry["document_duplicate_ids"] == 0, name


def test_query_ids_unique_per_dataset():
    d = _load()["datasets"]
    for name, entry in d.items():
        assert entry["query_unique_ids"] == entry["example_count_observed"], name
        assert entry["query_duplicate_ids"] == 0, name


def test_manifest_identities_exact():
    d = _load()["datasets"]
    for name, entry in d.items():
        assert entry["manifest"]["dataset"] == name, name