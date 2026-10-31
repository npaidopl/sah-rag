"""Offline tests for the LongRAG baseline adapter.

All tests run on CPU, offline, and complete in well under 10 seconds.
Native mode is exercised only when torch + transformers are available
AND an embedder path is configured (see ``_native_available``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from baselines.base import (
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)
from sah_rag.canonical.models import CanonicalDocument
from baselines.longrag_v2.adapter import LongRAGBaseline


# ---------------------------------------------------------------------------
# Deterministic mock embedder (no network, no tokenizer, no randomness)
# ---------------------------------------------------------------------------

class _MockKeywordEmbedder:
    """Bag-of-words embedder with a stateful round-robin vocabulary.

    Purpose: exercise the adapter's grouping, indexing, and ranking logic
    deterministically. NOT a real embedder; do not use outside tests.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self._vocab: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "mock-keyword"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for tok in text.lower().split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab) % self._dim
            v[self._vocab[tok]] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vec(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    doc_id: str, text: str, dataset: str = "mock"
) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset=dataset,
        text_raw=text,
        text_normalized=text,
    )


def _make_corpus() -> list[CanonicalDocument]:
    return [
        _make_doc("d1", "the cat sat on the mat"),
        _make_doc("d2", "dogs chase balls in the park"),
        _make_doc("d3", "rag systems retrieve relevant context"),
        _make_doc("d4", "long context documents need special handling"),
        _make_doc("d5", "legal contracts are structured hierarchically"),
    ]


def _assert_contract(
    results: list[RetrievedChunk],
    corpus_ids: set[str],
    k: int,
) -> None:
    assert isinstance(results, list)
    assert 0 <= len(results) <= k
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.doc_id in corpus_ids
        assert r.text
        assert isinstance(r.score, float)


def _native_available() -> bool:
    """Cheap availability check: spec lookups only, no actual imports."""
    if os.environ.get("LONGRAG_EMBEDDER_PATH") is None:
        return False
    import importlib.util
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_identity() -> None:
    assert LongRAGBaseline.name == "longrag"
    assert LongRAGBaseline.status == BaselineStatus.EXECUTABLE
    assert LongRAGBaseline.supports_native is True
    assert LongRAGBaseline.supports_controlled is True
    assert LongRAGBaseline.is_llm_based is False


def test_controlled_build_and_retrieve(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="longrag",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={
            "grouping_policy": "one_document_per_unit",
            "target_unit_tokens": 4096,
            "k1": 100,
        },
    )
    baseline = LongRAGBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=_MockKeywordEmbedder(),
    )
    baseline.build_index(_make_corpus())

    # Persistence artifacts must exist
    assert (tmp_path / "idx" / "unit_vectors.npy").exists()
    assert (tmp_path / "idx" / "unit_records.json").exists()
    assert (tmp_path / "idx" / "index_meta.json").exists()

    results = baseline.retrieve("cat mat", top_k=3)
    _assert_contract(results, {d.doc_id for d in _make_corpus()}, 3)
    # "cat" and "mat" both appear in d1; it must be rank 1.
    assert results[0].doc_id == "d1"


def test_adjacent_segments_multi_doc_expansion(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="longrag",
        mode=RunMode.CONTROLLED,
        top_k=5,
        extra={
            "grouping_policy": "adjacent_document_segments",
            "target_unit_tokens": 40,   # two units: {d1..d4} and {d5}
            "k1": 100,
        },
    )
    baseline = LongRAGBaseline(
        config=cfg,
        work_dir=tmp_path / "idx2",
        embedder=_MockKeywordEmbedder(),
    )
    baseline.build_index(_make_corpus())

    results = baseline.retrieve("cat mat", top_k=5)
    _assert_contract(results, {d.doc_id for d in _make_corpus()}, 5)
    # d1 lives in the top-scoring unit; it must be expanded and returned.
    doc_ids = {r.doc_id for r in results}
    assert "d1" in doc_ids
    # Ranks must still be contiguous even when expansion produces ties.
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_retrieve_before_build_raises(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="longrag",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={"grouping_policy": "one_document_per_unit"},
    )
    baseline = LongRAGBaseline(
        config=cfg,
        work_dir=tmp_path / "empty",
        embedder=_MockKeywordEmbedder(),
    )
    with pytest.raises(RuntimeError):
        baseline.retrieve("anything")


def test_unknown_grouping_policy_raises(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="longrag",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={"grouping_policy": "does_not_exist"},
    )
    baseline = LongRAGBaseline(
        config=cfg,
        work_dir=tmp_path / "bad",
        embedder=_MockKeywordEmbedder(),
    )
    with pytest.raises(ValueError):
        baseline.build_index(_make_corpus())


def test_empty_corpus_raises(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="longrag",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={"grouping_policy": "one_document_per_unit"},
    )
    baseline = LongRAGBaseline(
        config=cfg,
        work_dir=tmp_path / "emptycorpus",
        embedder=_MockKeywordEmbedder(),
    )
    with pytest.raises(ValueError):
        baseline.build_index([])


@pytest.mark.skipif(
    not _native_available(),
    reason="native mode requires torch + transformers + LONGRAG_EMBEDDER_PATH",
)
def test_native_smoke(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="longrag",
        mode=RunMode.NATIVE,
        top_k=3,
        extra={
            "grouping_policy": "one_document_per_unit",
            "target_unit_tokens": 4096,
            "k1": 100,
        },
    )
    baseline = LongRAGBaseline(config=cfg, work_dir=tmp_path / "native")
    baseline.build_index(_make_corpus())
    results = baseline.retrieve("cat mat", top_k=3)
    _assert_contract(results, {d.doc_id for d in _make_corpus()}, 3)