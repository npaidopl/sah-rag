"""Tests for the Step 91 real dataset sizing registry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "real_dataset_sizing_registry.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_real_dataset_sizing_registry.py"

EXPECTED = (
    "ruler",
    "loong",
    "long_bench_v2",
    "lara",
    "greek_bar_bench",
    "cuad",
    "clerk",
)

ALLOWED = {
    "INVENTORIED",
    "MISSING",
    "AMBIGUOUS_ROOT",
    "INVALID_JSON",
    "SCHEMA_INVALID",
    "REFERENCE_INVALID",
    "MANIFEST_MISMATCH",
}


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_registry():
    assert _run().returncode == 0


def test_seven_dataset_records():
    d = _load()["datasets"]
    for ds in EXPECTED:
        assert ds in d, ds


def test_status_vocabulary_enforced():
    d = _load()["datasets"]
    for ds in EXPECTED:
        assert d[ds]["status"] in ALLOWED


def test_totals_equal_sum_of_records():
    d = _load()["datasets"]
    t = _load()["portfolio_totals"]
    expected_docs = sum(
        s["documents"].get("record_count", 0)
        for s in d.values() if s["status"] == "INVENTORIED"
    )
    assert t["documents"] == expected_docs


def test_percentile_method_stable():
    assert _load()["measurement"]["percentile_method"] == "nearest_rank"


def test_dense_byte_formula_valid():
    d = _load()["datasets"]
    for s in d.values():
        if s["status"] != "INVENTORIED":
            continue
        sz = s["structural_sizing"]
        assert (
            sz["projected_dense_float32_bytes_minimum"]
            == sz["projected_embedding_records_minimum"] * 1024 * 4
        )
        assert (
            sz["projected_dense_float32_bytes_proxy_upper_bound"]
            == sz["projected_embedding_records_proxy_upper_bound"] * 1024 * 4
        )


def test_sparse_estimate_unavailable_without_model():
    assert _load()["measurement"]["model_loaded"] is False
    assert _load()["measurement"]["tokenizer_loaded"] is False