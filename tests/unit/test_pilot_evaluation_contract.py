"""Tests for the Step 88 evaluation contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PATH = REPO_ROOT / "baseline_audit" / "pilot_evaluation_contract.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pilot_evaluation_contract.py"


def _load() -> dict:
    return yaml.safe_load(PATH.read_text(encoding="utf-8"))


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_validator_accepts_contract():
    assert _run().returncode == 0


def test_evaluation_unit_canonical_document():
    assert _load()["evaluation"]["unit"] == "canonical_document"


def test_primary_method_is_hybrid_rrf():
    assert _load()["evaluation"]["primary_retrieval_method"] == "hybrid_rrf"


def test_cutoffs_exact():
    assert list(_load()["evaluation"]["cutoffs"]) == [1, 5, 10, 100]


def test_execution_disabled():
    exec_ = _load()["execution"]
    for flag in (
        "runtime_judgment_loading_enabled",
        "evaluation_enabled",
        "metric_computation_enabled",
        "generation_enabled",
        "answer_evaluation_enabled",
        "llm_as_judge_enabled",
    ):
        assert exec_[flag] is False, flag


def test_generation_and_answer_evaluation_disabled():
    exec_ = _load()["execution"]
    assert exec_["generation_enabled"] is False
    assert exec_["answer_evaluation_enabled"] is False
    assert exec_["llm_as_judge_enabled"] is False