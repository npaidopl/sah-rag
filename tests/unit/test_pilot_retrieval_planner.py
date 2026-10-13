"""Tests for the Step 86 hybrid retrieval planner helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER = REPO_ROOT / "scripts" / "plan_pilot_hybrid_retrieval.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("_pilot_retrieval_planner", PLANNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_query_plan_hash_deterministic():
    planner = _load_planner()
    queries = [
        {"query_id": "pilot-query-01", "query_text": "alpha"},
        {"query_id": "pilot-query-02", "query_text": "beta"},
    ]
    h1 = planner.compute_query_plan_hash(queries)
    h2 = planner.compute_query_plan_hash(queries)
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)


def test_retrieval_plan_hash_deterministic():
    planner = _load_planner()
    h1 = planner.compute_retrieval_plan_hash("a" * 64)
    h2 = planner.compute_retrieval_plan_hash("a" * 64)
    assert h1 == h2
    assert len(h1) == 64


def test_planner_does_not_load_model_or_open_qdrant():
    source = PLANNER.read_text(encoding="utf-8")
    assert "BGEM3FlagModel" not in source
    assert "QdrantClient(" not in source


def test_planner_does_not_use_judgments():
    source = PLANNER.read_text(encoding="utf-8")
    for forbidden in ("reference_answer", "relevance", "judgment", "relevance_label"):
        assert forbidden not in source, forbidden


def test_planner_does_not_create_retrieval_root():
    source = PLANNER.read_text(encoding="utf-8")
    assert ".mkdir(" not in source