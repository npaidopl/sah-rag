"""Verify (Ξ²) controlled-adaptation structural equivalence for BM25.

Claim (ARM_REGISTRY Β§2): for BM25, native and controlled mode produce
identical retrieval output, because BM25 is lexical and non-LLM-based.
The adapter accepts injected components per the frozen BaseBaseline
contract but structurally ignores them.

This test proves the claim empirically.
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


class _DummyEmbedder:
    """Minimal Embedder Protocol implementation.

    BM25 structurally ignores this. Present only to satisfy the frozen
    BaseBaseline._validate() check for controlled mode.
    """
    @property
    def name(self) -> str:
        return "dummy-embedder-not-used-by-bm25"

    def encode_documents(self, texts):
        raise AssertionError("BM25 must not call encode_documents")

    def encode_query(self, text):
        raise AssertionError("BM25 must not call encode_query")


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
def test_controlled_equivalent_to_native(tmp_path: Path) -> None:
    docs = [
        _make_doc("d1", "The cat sat on the mat near the window."),
        _make_doc("d2", "Dogs chase balls in the park every morning."),
        _make_doc("d3", "RAG systems retrieve relevant context for generation."),
        _make_doc("d4", "Long-context documents need special handling strategies."),
        _make_doc("d5", "Legal contracts are structured hierarchically."),
    ]
    queries = [
        "cat on the mat",
        "retrieve relevant context",
        "legal contracts hierarchical",
    ]

    cfg_native = BaselineConfig(name="bm25", mode=RunMode.NATIVE, top_k=10)
    native = BM25Baseline(config=cfg_native, work_dir=tmp_path / "native")
    native.build_index(docs)
    native_results = {q: native.retrieve(q, top_k=10) for q in queries}

    cfg_controlled = BaselineConfig(name="bm25", mode=RunMode.CONTROLLED, top_k=10)
    controlled = BM25Baseline(
        config=cfg_controlled,
        work_dir=tmp_path / "controlled",
        embedder=_DummyEmbedder(),
    )
    controlled.build_index(docs)
    controlled_results = {q: controlled.retrieve(q, top_k=10) for q in queries}

    for q in queries:
        n = native_results[q]
        c = controlled_results[q]
        assert [r.doc_id for r in n] == [r.doc_id for r in c], (
            f"Query {q!r}: doc_ids differ between native and controlled."
        )
        assert [r.rank for r in n] == [r.rank for r in c]
        assert [round(r.score, 6) for r in n] == [round(r.score, 6) for r in c]


@pytest.mark.skipif(
    not (JAVA_AVAILABLE and PYSERINI_AVAILABLE),
    reason=SKIP_REASON,
)
def test_controlled_mode_requires_embedder(tmp_path: Path) -> None:
    """Verify contract validation rejects controlled mode without embedder."""
    cfg = BaselineConfig(name="bm25", mode=RunMode.CONTROLLED, top_k=10)
    with pytest.raises(ValueError, match="controlled mode requires embedder"):
        BM25Baseline(config=cfg, work_dir=tmp_path / "no_embedder")