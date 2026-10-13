"""Unit tests for SAH-RAG M4B reduce."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_sah_rag_m4b_reduce.py"


def _load():
    spec = importlib.util.spec_from_file_location("m4b", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m4b"] = m
    spec.loader.exec_module(m)
    return m


def test_monotonicity_empty():
    m = _load()
    assert m.check_monotonicity([])["verdict"] == "INSUFFICIENT_DATA"


def test_monotonicity_increasing():
    m = _load()
    curve = [
        {"value": 1, "recall@10": 0.1, "hit@10": 0.1, "ndcg@10": 0.1,
         "mrr@10": 0.1, "precision@10": 0.1},
        {"value": 2, "recall@10": 0.2, "hit@10": 0.2, "ndcg@10": 0.2,
         "mrr@10": 0.2, "precision@10": 0.2},
        {"value": 3, "recall@10": 0.3, "hit@10": 0.3, "ndcg@10": 0.3,
         "mrr@10": 0.3, "precision@10": 0.3},
    ]
    v = m.check_monotonicity(curve)
    assert v["verdict"] == "MONOTONE"
    assert v["direction"] == "increasing"


def test_monotonicity_non_monotone():
    m = _load()
    curve = [
        {"value": 1, "recall@10": 0.1, "hit@10": 0.1, "ndcg@10": 0.1,
         "mrr@10": 0.1, "precision@10": 0.1},
        {"value": 2, "recall@10": 0.3, "hit@10": 0.3, "ndcg@10": 0.3,
         "mrr@10": 0.3, "precision@10": 0.3},
        {"value": 3, "recall@10": 0.2, "hit@10": 0.2, "ndcg@10": 0.2,
         "mrr@10": 0.2, "precision@10": 0.2},
    ]
    v = m.check_monotonicity(curve)
    assert v["verdict"] == "NON_MONOTONE"


def test_saturation():
    m = _load()
    curve = [
        {"value": 1, "recall@10": 0.1, "hit@10": 0.1, "ndcg@10": 0.1,
         "mrr@10": 0.1, "precision@10": 0.1},
        {"value": 2, "recall@10": 0.2, "hit@10": 0.2, "ndcg@10": 0.2,
         "mrr@10": 0.2, "precision@10": 0.2},
        {"value": 3, "recall@10": 0.205, "hit@10": 0.2, "ndcg@10": 0.2,
         "mrr@10": 0.2, "precision@10": 0.2},
    ]
    v = m.find_saturation(curve, threshold=0.01)
    assert v["verdict"] == "SATURATED"
    assert v["saturation_value"] == 2