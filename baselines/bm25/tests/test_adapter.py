"""Offline CPU test for the BM25 baseline adapter.

Runs the full build_index -> retrieve flow on a 5-document corpus.
Skips if Java is not available (Pyserini/Anserini requires a JVM).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from baselines.base import BaselineConfig, RunMode
from sah_rag.canonical.models import CanonicalDocument
from baselines.bm25.adapter import BM25Baseline

JAVA_AVAILABLE = shutil.which("java") is not None
PYSERINI_AVAILABLE = True
try:
    import pyserini  # noqa: F401
except Exception:
    PYSERINI_AVAILABLE = False

SKIP_REASON = "Java 21 + pyserini required for BM25 (Pyserini/Anserini)."


def _make_doc(doc_id: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset="unit-test",
        text_raw=text,
        text_normalized=text,
    )


@pytest.mark.skipif(
    not (JAVA_AVAILABLE and PYSERINI_AVAILABLE),
    reason=SKIP_REASON,
)
def test_build_and_retrieve(tmp_path: Path) -> None:
    docs = [
        _make_doc("d1", "The cat sat on the mat near the window."),
        _make_doc("d2", "Dogs chase balls in the park every morning."),
        _make_doc("d3", "RAG systems retrieve relevant context for generation."),
        _make_doc("d4", "Long-context documents need special handling strategies."),
        _make_doc("d5", "Legal contracts are structured hierarchically."),
    ]

    cfg = BaselineConfig(name="bm25", mode=RunMode.NATIVE, top_k=3)
    baseline = BM25Baseline(config=cfg, work_dir=tmp_path / "idx")

    baseline.build_index(docs)
    assert (tmp_path / "idx" / "lucene_index").exists(), "Lucene index dir missing"

    queries = [
        ("cat on the mat", 3),
        ("retrieve relevant context", 3),
        ("legal contracts hierarchical", 2),
    ]
    corpus_ids = {d.doc_id for d in docs}

    for q, k in queries:
        results = baseline.retrieve(q, top_k=k)
        assert isinstance(results, list)
        assert len(results) <= k
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        for r in results:
            assert r.doc_id in corpus_ids, f"unknown doc_id: {r.doc_id}"
            assert r.text, "retrieved text must be non-empty"
            assert r.chunk_id == r.doc_id, "BM25 chunk_id must equal doc_id"
            assert r.score >= 0.0