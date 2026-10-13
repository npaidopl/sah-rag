"""Tests for the Step 87 retrieval execution contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_retrieval_execution_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_retrieval_execution_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_execution_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_query_embedding_and_retrieval_enabled():
    exec_ = _load()["execution"]
    for flag in (
        "query_embedding_enabled",
        "dense_retrieval_enabled",
        "sparse_retrieval_enabled",
        "hybrid_retrieval_enabled",
        "parent_aggregation_enabled",
        "document_expansion_enabled",
    ):
        assert exec_[flag] is True, flag


def test_evaluation_and_generation_disabled():
    exec_ = _load()["execution"]
    assert exec_["evaluation_enabled"] is False
    assert exec_["generation_enabled"] is False
    assert exec_["reranking_enabled"] is False


def test_query_count_exactly_six():
    assert _load()["source"]["expected_query_count"] == 6


def test_collection_name_exact():
    assert (
        _load()["collection"]["physical_name"]
        == "rag_infrastructure_pilot_synthetic_v1_13d22ed4f1f0"
    )


def test_judgments_access_forbidden():
    source = _load()["source"]
    assert source["judgments_path_access_allowed"] is False
    assert source["reference_answers_access_allowed"] is False