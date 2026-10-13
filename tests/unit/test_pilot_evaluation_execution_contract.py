"""Tests for the Step 89 evaluation execution contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_evaluation_execution_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_evaluation_execution_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_contract():
    assert _run().returncode == 0


def test_evaluation_and_metric_computation_enabled():
    e = _load()["execution"]
    for flag in (
        "runtime_judgment_loading_enabled",
        "retrieval_case_construction_enabled",
        "evaluation_enabled",
        "metric_computation_enabled",
        "aggregate_metric_computation_enabled",
        "persistence_enabled",
    ):
        assert e[flag] is True, flag


def test_retrieval_generation_answer_llm_disabled():
    e = _load()["execution"]
    for flag in (
        "query_embedding_enabled",
        "retrieval_enabled",
        "qdrant_client_enabled",
        "architecture_comparison_enabled",
        "generation_enabled",
        "answer_evaluation_enabled",
        "llm_as_judge_enabled",
    ):
        assert e[flag] is False, flag


def test_six_queries_twelve_judgments_frozen():
    s = _load()["source"]
    assert s["expected_query_count"] == 6
    assert s["expected_judgment_count"] == 12
    assert s["expected_relevant_documents_per_query"] == 2


def test_cutoffs_exact():
    assert list(_load()["evaluation"]["cutoffs"]) == [1, 5, 10, 100]


def test_expected_counts_typed_row_model():
    c = _load()["expected_counts"]
    assert c["retrieval_case_count"] == 6
    assert c["query_metric_rows"] == 24
    assert c["aggregate_metric_rows"] == 4