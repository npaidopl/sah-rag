"""Tests for the Step 88 evaluation planner helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER = REPO_ROOT / "scripts" / "plan_pilot_retrieval_evaluation.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("_pilot_eval_planner", PLANNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_judgment_plan_hash_deterministic():
    planner = _load_planner()
    judgments = {"q-1": ["d-1"], "q-2": ["d-2"]}
    qids = ["q-1", "q-2"]
    h1 = planner.compute_judgment_plan_hash(judgments, qids)
    h2 = planner.compute_judgment_plan_hash(judgments, qids)
    assert h1 == h2
    assert len(h1) == 64


def test_evaluation_plan_hash_deterministic():
    planner = _load_planner()
    h1 = planner.compute_evaluation_plan_hash("a" * 64, ["ndcg_at_k"])
    h2 = planner.compute_evaluation_plan_hash("a" * 64, ["ndcg_at_k"])
    assert h1 == h2
    assert len(h1) == 64


def test_metric_inventory_matches_common_contract():
    planner = _load_planner()
    ec = planner.validate_dependencies()
    metrics = set(ec["metric_names"])
    assert metrics == {
        "hit_at_k",
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
    }


def test_planner_does_not_call_evaluator():
    source = PLANNER.read_text(encoding="utf-8")
    assert "evaluate_case(" not in source
    assert "evaluate_retrieval_run(" not in source


def test_planner_does_not_create_evaluation_root():
    source = PLANNER.read_text(encoding="utf-8")
    # Planner may check .exists() but must not call .mkdir on the eval root
    assert "evaluation_root.mkdir" not in source
    assert "evaluation_root.touch" not in source


def test_no_metrics_are_persisted():
    source = PLANNER.read_text(encoding="utf-8")
    for forbidden in ("query_metrics.jsonl", "aggregate_metrics.json", "retrieval_cases.jsonl"):
        assert forbidden not in source or "planned_files" in source