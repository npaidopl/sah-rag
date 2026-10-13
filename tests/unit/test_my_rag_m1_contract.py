"""Unit tests for My_RAG M1 optimization contract freeze."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_my_rag_m1_freeze.py"


def _load():
    spec = importlib.util.spec_from_file_location("m1", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m1"] = m
    spec.loader.exec_module(m)
    return m


def test_split_deterministic():
    m = _load()
    s1 = m.build_split("clerk", seed=0, dev_fraction=0.5)
    s2 = m.build_split("clerk", seed=0, dev_fraction=0.5)
    assert s1["dev_query_ids"] == s2["dev_query_ids"]
    assert s1["test_query_ids"] == s2["test_query_ids"]
    assert s1["dev_doc_ids"] == s2["dev_doc_ids"]


def test_split_no_query_leakage():
    m = _load()
    for ds in ("clerk", "cuad", "greek_bar_bench", "lara"):
        s = m.build_split(ds, seed=0, dev_fraction=0.5)
        dev = set(s["dev_query_ids"])
        test = set(s["test_query_ids"])
        assert dev.isdisjoint(test), f"{ds}: query leakage"


def test_split_no_doc_leakage():
    m = _load()
    for ds in ("clerk", "cuad", "greek_bar_bench", "lara"):
        s = m.build_split(ds, seed=0, dev_fraction=0.5)
        dev_docs = set(s["dev_doc_ids"])
        test_docs = set(s["test_doc_ids"])
        assert dev_docs.isdisjoint(test_docs), f"{ds}: doc leakage"


def test_split_full_coverage():
    m = _load()
    for ds in ("clerk", "cuad", "greek_bar_bench", "lara"):
        s = m.build_split(ds, seed=0, dev_fraction=0.5)
        assert (
            len(s["dev_query_ids"]) + len(s["test_query_ids"])
            == s["total_queries"]
        ), f"{ds}: not full coverage"


def test_chunker_config_exposed():
    m = _load()
    d = m.discover_parameters()
    assert d.get("chunker_config_ok") is True
    assert "chunk_size" in d.get("chunker_defaults", {})
    assert d["chunker_defaults"]["chunk_size"] == 256


def test_chunker_overrides_accepted():
    m = _load()
    d = m.discover_parameters()
    probes = d.get("chunker_override_ok", {})
    for key in ("512_32", "1024_128", "256_0"):
        assert probes.get(key) is True, f"override {key} rejected"


def test_retrieval_signatures_have_limits():
    m = _load()
    d = m.discover_parameters()
    assert "limit" in d.get("retrieve_dense_params", [])
    assert "limit" in d.get("retrieve_sparse_params", [])
    assert "candidate_limit" in d.get("retrieve_hybrid_rrf_params", [])
    assert "result_limit" in d.get("retrieve_hybrid_rrf_params", [])