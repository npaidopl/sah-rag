"""Unit tests for My_RAG M7 locked test."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_my_rag_m7_locked_test.py"


def _load():
    spec = importlib.util.spec_from_file_location("m7", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m7"] = m
    spec.loader.exec_module(m)
    return m


def test_frozen_config_exists():
    m = _load()
    cfg = m.load_frozen_config()
    assert "selected_config_id" in cfg
    assert cfg["selected_config_id"] == "c256_o32_hybrid_rrf_k200"


def test_frozen_config_values():
    m = _load()
    cfg = m.load_frozen_config()
    sel = cfg["selected_config"]
    assert sel["chunk_size"] == 256
    assert sel["chunk_overlap"] == 32
    assert sel["retrieval_mode"] == "hybrid_rrf"
    assert sel["candidate_limit"] == 200


def test_metrics_in_range():
    m = _load()
    ranked = ["a", "b", "c"]
    rel = {"b", "d"}
    assert 0.0 <= m.hit_at_k(ranked, rel, 3) <= 1.0
    assert 0.0 <= m.recall_at_k(ranked, rel, 3) <= 1.0
    assert 0.0 <= m.precision_at_k(ranked, rel, 3) <= 1.0
    assert 0.0 <= m.reciprocal_rank(ranked, rel, 3) <= 1.0
    assert 0.0 <= m.ndcg_at_k(ranked, rel, 3) <= 1.0