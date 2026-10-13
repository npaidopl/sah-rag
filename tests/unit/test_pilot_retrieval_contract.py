"""Tests for the Step 86 hybrid retrieval contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_retrieval_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_retrieval_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_validator_accepts_retrieval_contract():
    r = _run()
    assert r.returncode == 0, r.stderr


def test_six_query_count_frozen():
    assert _load()["source"]["expected_query_count"] == 6


def test_primary_method_is_hybrid_rrf():
    assert _load()["retrieval"]["primary_method"] == "hybrid_rrf"


def test_limits_are_exactly_twelve():
    ret = _load()["retrieval"]
    assert ret["candidate_limit"] == 12
    assert ret["child_result_limit"] == 12
    assert ret["parent_result_limit"] == 12
    assert ret["document_result_limit"] == 12


def test_reranker_disabled():
    assert _load()["retrieval"]["reranker_enabled"] is False


def test_all_execution_flags_false():
    execution = _load()["execution"]
    for flag in (
        "query_embedding_enabled",
        "dense_retrieval_enabled",
        "sparse_retrieval_enabled",
        "hybrid_retrieval_enabled",
        "parent_aggregation_enabled",
        "document_expansion_enabled",
        "evaluation_enabled",
        "generation_enabled",
    ):
        assert execution[flag] is False, flag