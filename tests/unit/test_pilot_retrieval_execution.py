"""Tests for the Step 87 retrieval execution runner helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_pilot_hybrid_retrieval.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_pilot_retrieval_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_does_not_reference_judgments_or_reference_answers_in_runtime():
    """Runtime module must not open judgments.jsonl or read reference answers.

    The only permitted occurrences are static string constants that
    explicitly mark these artifacts as prohibited / not loaded.
    """
    source = RUNNER.read_text(encoding="utf-8")
    assert "judgments.jsonl" not in source
    assert "reference_answer\"" not in source or "reference_answers_access_allowed" in source


def test_runner_does_not_import_evaluation_module():
    source = RUNNER.read_text(encoding="utf-8")
    assert "evaluation" not in source


def test_runner_does_not_call_retrieval_evaluation():
    source = RUNNER.read_text(encoding="utf-8")
    for fn in ("retrieve_hybrid_parents",):
        assert fn not in source, fn


def test_query_projection_keeps_only_id_and_text():
    """load_queries must return dicts with only query_id and query_text.

    We invoke the function against the real contract and assert the
    projected record keys directly, rather than grepping source.
    """
    runner = _load_runner()
    from pathlib import Path

    import yaml

    contract = yaml.safe_load(
        (
            Path(runner.__file__).resolve().parents[1]
            / "baseline_audit"
            / "pilot_retrieval_execution_contract.yaml"
        ).read_text(encoding="utf-8")
    )
    queries = runner.load_queries(contract)
    assert len(queries) == 6
    for q in queries:
        assert set(q.keys()) == {"query_id", "query_text"}
        assert q["query_text"].strip() != ""