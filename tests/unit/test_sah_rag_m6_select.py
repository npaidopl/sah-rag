"""Unit tests for SAH-RAG M6 selection."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_sah_rag_m6_select.py"


def _load():
    spec = importlib.util.spec_from_file_location("m6", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m6"] = m
    spec.loader.exec_module(m)
    return m


def test_parse_config_id():
    m = _load()
    d = m.parse_config_id("c256_o128_hybrid_rrf_k200")
    assert d["chunk_size"] == 256
    assert d["chunk_overlap"] == 128
    assert d["retrieval_mode"] == "hybrid_rrf"
    assert d["candidate_limit"] == 200


def test_parse_config_id_dense():
    m = _load()
    d = m.parse_config_id("c512_o0_dense_k100")
    assert d["retrieval_mode"] == "dense"


def test_lexicographic_order():
    m = _load()
    # valid config_id format
    ca = "c256_o32_hybrid_rrf_k200"   # lat=0.01, storage=100
    cb = "c512_o32_hybrid_rrf_k200"   # lat=0.02, storage=50
    cc = "c256_o128_hybrid_rrf_k200"  # lat=0.01, storage=200
    latency = {ca: 0.01, cb: 0.02, cc: 0.01}
    storage = {ca: 100.0, cb: 50.0, cc: 200.0}
    out = m.lexicographic_tiebreak([ca, cb, cc], latency, storage)
    # latency asc: ca & cc tie -> storage asc: ca (100) < cc (200)
    # then cb
    assert out == [ca, cc, cb]


def test_pareto_simple():
    m = _load()
    primary = {"a": 0.5, "b": 0.4, "c": 0.4}
    latency = {"a": 0.1, "b": 0.05, "c": 0.2}
    storage = {"a": 100.0, "b": 50.0, "c": 200.0}
    # a: (0.5, 0.1, 100)
    # b: (0.4, 0.05, 50)
    # c: (0.4, 0.2, 200)
    # b dominates c (higher primary equal? same; lower latency; lower storage)
    # neither a nor b dominates the other
    frontier = m.pareto_frontier(["a", "b", "c"], primary, latency, storage)
    assert "a" in frontier
    assert "b" in frontier
    assert "c" not in frontier