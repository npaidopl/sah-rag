"""Tests for the Step 93 dataset retrieval task matrix."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "dataset_retrieval_task_matrix.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_dataset_retrieval_task_matrix.py"

DATASETS = ("ruler","loong","long_bench_v2","lara","greek_bar_bench","cuad","clerk")


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_matrix():
    assert _run().returncode == 0


def test_seven_dataset_records():
    assert set(_load()["datasets"].keys()) == set(DATASETS)


def test_supplied_context_not_embedded():
    for name, r in _load()["datasets"].items():
        assert r["query_text_projection"]["supplied_context_in_query_embedding"] is False, name


def test_cuad_zero_relevance_policy():
    r = _load()["datasets"]["cuad"]
    assert r["zero_relevance_policy"] == "EXPLICIT_NEGATIVE_CASE"
    assert r["zero_relevance_count_expected"] == 14208
    assert r["positive_evidence_count_expected"] == 6702


def test_cuad_subsets_separated():
    cp = _load()["cuad_evaluation_policy"]
    assert cp["combined_reporting"]["separate_subsets_required"] is True
    assert cp["combined_reporting"]["macro_mix_allowed"] is False


def test_rag_system_tuning_policy():
    t = _load()["rag_system_tuning"]
    assert t["test_data_for_tuning_prohibited"] is True
    assert t["selected_configuration_frozen_before_test"] is True


def test_every_dataset_has_leakage_prohibitions():
    for name, r in _load()["datasets"].items():
        assert len(r["leakage_prohibitions"]) >= 1, name