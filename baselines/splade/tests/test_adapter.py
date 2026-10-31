"""Offline tests for SPLADE adapter.

All tests run offline, on CPU, in <10s.
Native mode is tested with `_encode_native` monkeypatched (no torch,
no network, no model download).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from baselines.base import (
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)
from sah_rag.canonical.models import CanonicalDocument
from baselines.splade.adapter import SpladeBaseline


def _stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


class FakeEmbedder:
    """Deterministic dense embedder for controlled-mode tests."""

    name = "fake-bge-m3"
    dim = 32

    def _embed(self, text: str) -> list[float]:
        rng = np.random.default_rng(_stable_seed(text))
        return rng.standard_normal(self.dim).tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeSparseEmbedder:
    """Fake embedder that exposes encode_sparse -> dict[token_id -> weight]."""

    name = "fake-sparse-bge-m3"

    def _sparse(self, text: str) -> dict[int, float]:
        rng = np.random.default_rng(_stable_seed(text))
        idx = rng.choice(1000, size=16, replace=False)
        return {int(i): float(rng.random()) for i in idx}

    def encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return [self._sparse(t) for t in texts]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        # Dense fallback (never used in sparse-capable tests)
        rng = np.random.default_rng(0)
        return [rng.standard_normal(32).tolist() for _ in texts]

    def encode_query(self, text: str) -> list[float]:
        return np.zeros(32).tolist()


def make_doc(doc_id: str, text: str, dataset: str = "mock") -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset=dataset,
        text_raw=text,
        text_normalized=text,
    )


MOCK_CORPUS = [
    make_doc("d1", "The cat sat on the mat."),
    make_doc("d2", "Dogs chase balls in the park."),
    make_doc("d3", "RAG systems retrieve relevant context."),
    make_doc("d4", "Long-context documents need special handling."),
    make_doc("d5", "Legal contracts are structured hierarchically."),
]


# --------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------- #
def test_identity_attributes() -> None:
    assert SpladeBaseline.name == "splade"
    assert SpladeBaseline.version == "0.1.0"
    assert SpladeBaseline.supports_native is True
    assert SpladeBaseline.supports_controlled is True
    assert SpladeBaseline.is_llm_based is False
    assert SpladeBaseline.status == BaselineStatus.EXECUTABLE


# --------------------------------------------------------------------- #
# Controlled mode
# --------------------------------------------------------------------- #
def test_controlled_build_and_retrieve(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="splade",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={"sparse_top_m": 8},
    )
    bl = SpladeBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=FakeEmbedder(),
    )
    bl.build_index(MOCK_CORPUS)
    assert (tmp_path / "idx").exists()
    assert (tmp_path / "idx" / "index.json").exists()
    assert (tmp_path / "idx" / "doc_meta.json").exists()

    results = bl.retrieve("RAG retrieval", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.doc_id in {d.doc_id for d in MOCK_CORPUS}
        assert r.text
        assert isinstance(r.score, float)


def test_controlled_empty_index_returns_empty(tmp_path: Path) -> None:
    cfg = BaselineConfig(name="splade", mode=RunMode.CONTROLLED, top_k=3)
    bl = SpladeBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=FakeEmbedder(),
    )
    assert bl.retrieve("anything") == []


def test_controlled_contiguous_ranks(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="splade",
        mode=RunMode.CONTROLLED,
        top_k=10,
        extra={"sparse_top_m": 8},
    )
    bl = SpladeBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=FakeEmbedder(),
    )
    bl.build_index(MOCK_CORPUS)
    results = bl.retrieve("documents", top_k=4)
    assert len(results) <= 4
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_controlled_requires_embedder(tmp_path: Path) -> None:
    cfg = BaselineConfig(name="splade", mode=RunMode.CONTROLLED, top_k=3)
    with pytest.raises(ValueError):
        SpladeBaseline(config=cfg, work_dir=tmp_path / "x")


# --------------------------------------------------------------------- #
# Native mode (mocked, no torch)
# --------------------------------------------------------------------- #
def test_native_build_and_retrieve_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native mode with SPLADE encoder stubbed out (no torch, no network)."""

    def fake_encode_native(self, text: str) -> dict[int, float]:
        rng = np.random.default_rng(_stable_seed(text))
        idx = rng.choice(1000, size=16, replace=False)
        return {int(i): float(rng.random()) for i in idx}

    monkeypatch.setattr(SpladeBaseline, "_encode_native", fake_encode_native)
    monkeypatch.setattr(SpladeBaseline, "_load_native_encoder", lambda self: None)

    cfg = BaselineConfig(name="splade", mode=RunMode.NATIVE, top_k=3)
    bl = SpladeBaseline(config=cfg, work_dir=tmp_path / "idx")
    bl.build_index(MOCK_CORPUS)

    results = bl.retrieve("RAG retrieval", top_k=3)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert r.doc_id in {d.doc_id for d in MOCK_CORPUS}
        assert r.text


def test_native_empty_index_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SpladeBaseline, "_load_native_encoder", lambda self: None)
    cfg = BaselineConfig(name="splade", mode=RunMode.NATIVE, top_k=3)
    bl = SpladeBaseline(config=cfg, work_dir=tmp_path / "idx")
    assert bl.retrieve("anything") == []


# --------------------------------------------------------------------- #
# Persistence round-trip
# --------------------------------------------------------------------- #
def test_persistence_round_trip(tmp_path: Path) -> None:
    work_dir = tmp_path / "idx"
    cfg = BaselineConfig(
        name="splade",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={"sparse_top_m": 8},
    )
    bl1 = SpladeBaseline(
        config=cfg, work_dir=work_dir, embedder=FakeEmbedder()
    )
    bl1.build_index(MOCK_CORPUS)
    results1 = bl1.retrieve("RAG retrieval", top_k=3)

    bl2 = SpladeBaseline(
        config=cfg, work_dir=work_dir, embedder=FakeEmbedder()
    )
    results2 = bl2.retrieve("RAG retrieval", top_k=3)

    assert [r.doc_id for r in results1] == [r.doc_id for r in results2]
    assert [r.rank for r in results2] == list(range(1, len(results2) + 1))

# --------------------------------------------------------------------- #
# Sparse duck-typing (controlled mode enhancement)
# --------------------------------------------------------------------- #
def test_controlled_uses_sparse_when_available(tmp_path: Path) -> None:
    """When embedder exposes encode_sparse, adapter uses sparse dims (>=32)."""
    cfg = BaselineConfig(
        name="splade", mode=RunMode.CONTROLLED, top_k=3,
    )
    bl = SpladeBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=FakeSparseEmbedder(),
    )
    bl.build_index(MOCK_CORPUS)

    all_dims: set[int] = set()
    for sparse in bl._index.values():
        all_dims.update(sparse.keys())
    assert any(d >= 32 for d in all_dims), (
        "should use sparse dims (token ids up to 1000), not dense dims 0..31"
    )

    results = bl.retrieve("test query", top_k=3)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert r.doc_id in {d.doc_id for d in MOCK_CORPUS}
        assert r.text

