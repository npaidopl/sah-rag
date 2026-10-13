"""Unit tests for SAH-RAG M4A screening."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_sah_rag_m4a_screening.py"


def _load():
    spec = importlib.util.spec_from_file_location("m4a", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["m4a"] = m
    spec.loader.exec_module(m)
    return m


def test_grid_x_modes_x_candidates():
    m = _load()
    total = len(m.GRID) * len(m.MODES) * len(m.CANDIDATE_LIMITS)
    assert total == 81


def test_sample_subset_deterministic():
    m = _load()
    ids = [f"q_{i:04d}" for i in range(1000)]
    s1 = m.sample_dev_subset(ids, seed=42)
    s2 = m.sample_dev_subset(ids, seed=42)
    assert s1 == s2


def test_sample_subset_respects_bounds():
    m = _load()
    ids = [f"q_{i:04d}" for i in range(2000)]
    s = m.sample_dev_subset(ids)
    assert len(s) >= m.SAMPLE_MIN
    assert len(s) <= m.SAMPLE_MAX


def test_rrf_fuse_monotone():
    m = _load()
    fused = m.rrf_fuse([10, 20, 30], [30, 40, 50])
    # 30 appears in both -> highest score
    assert fused[0][0] == 30
    # scores are decreasing
    scores = [s for _, s in fused]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_aggregate_empty():
    m = _load()
    out = m.aggregate_to_documents(
        [], chunk_ids=[], parent_ids=[], doc_ids=[],
    )
    assert out == []


def test_batch_topk_shapes():
    m = _load()
    scores = np.random.rand(5, 100).astype(np.float32)
    top = m.batch_topk(scores, 10)
    assert top.shape == (5, 10)
    # top-1 is argmax per row
    for i in range(5):
        assert top[i, 0] == int(np.argmax(scores[i]))