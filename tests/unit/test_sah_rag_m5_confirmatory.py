"""Unit tests for SAH-RAG M5 confirmatory."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_sah_rag_m5_confirmatory.py"


def _load():
    spec = importlib.util.spec_from_file_location("m5", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m5"] = m
    spec.loader.exec_module(m)
    return m


def test_confirmatory_size():
    m = _load()
    assert len(m.CONFIRMATORY) <= 8


def test_confirmatory_includes_default():
    m = _load()
    assert (256, 32, "hybrid_rrf", 200) in m.CONFIRMATORY


def test_confirmatory_includes_best_primary():
    m = _load()
    assert (256, 128, "hybrid_rrf", 200) in m.CONFIRMATORY


def test_config_id_format():
    m = _load()
    assert m.config_id(256, 32, "hybrid_rrf", 200) == "c256_o32_hybrid_rrf_k200"