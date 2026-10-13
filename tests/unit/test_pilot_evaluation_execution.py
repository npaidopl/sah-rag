"""Tests for the Step 89 evaluation execution runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_pilot_retrieval_evaluation.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_pilot_eval_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FORBIDDEN_REFERENCES = (
    "QdrantClient",
    "BGEM3FlagModel",
    "retrieve_dense",
    "retrieve_sparse",
    "retrieve_hybrid_rrf",
    "retrieve_hybrid_parents",
    "OpenAI",
    "generate_answer",
    "llm_as_judge(",
)


def test_runner_has_no_retrieval_or_model_references():
    source = RUNNER.read_text(encoding="utf-8")
    for token in FORBIDDEN_REFERENCES:
        assert token not in source, token


def test_runner_calls_evaluate_retrieval_run():
    source = RUNNER.read_text(encoding="utf-8")
    assert "evaluate_retrieval_run(" in source


def test_runner_does_not_call_evaluate_case_directly():
    source = RUNNER.read_text(encoding="utf-8")
    # Only evaluate_retrieval_run is permitted
    assert "evaluate_case(" not in source