"""Offline tests for the MemoRAG adapter.

These tests mock the `memorag` package so they run without GPU, network,
or the heavy dependencies. They verify contract compliance only.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from baselines.base import (
    BaselineConfig,
    RetrievedChunk,
    RunMode,
)
from sah_rag.canonical.models import CanonicalDocument
from baselines.memorag.adapter import MemoRAGBaseline


def _make_doc(doc_id: str, text: str, dataset: str = "mock") -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset=dataset,
        text_raw=text,
        text_normalized=text,
    )


def _install_fake_memorag(monkeypatch) -> None:
    class FakeMemoRAG:
        def __init__(self, **kwargs) -> None:
            self._docs: list[str] = []

        def load(self, docs) -> None:
            self._docs = list(docs)

        def __call__(self, query, task_type=None, top_k=10):
            n = min(top_k, len(self._docs))
            return [
                {
                    "doc_id": f"d{i + 1}",
                    "text": self._docs[i],
                    "score": 1.0 - i * 0.1,
                }
                for i in range(n)
            ]

    mod = types.ModuleType("memorag")
    mod.MemoRAG = FakeMemoRAG
    monkeypatch.setitem(sys.modules, "memorag", mod)


def _make_corpus() -> list[CanonicalDocument]:
    return [
        _make_doc("d1", "The cat sat on the mat."),
        _make_doc("d2", "Dogs chase balls in the park."),
        _make_doc("d3", "RAG systems retrieve relevant context."),
        _make_doc("d4", "Long-context documents need special handling."),
        _make_doc("d5", "Legal contracts are structured hierarchically."),
    ]


def test_build_and_retrieve_native(tmp_path: Path, monkeypatch) -> None:
    _install_fake_memorag(monkeypatch)
    docs = _make_corpus()
    cfg = BaselineConfig(name="memorag", mode=RunMode.NATIVE, top_k=3)
    baseline = MemoRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    baseline.build_index(docs)
    assert (tmp_path / "idx").exists()

    results = baseline.retrieve("RAG retrieval", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.doc_id in {d.doc_id for d in docs}
        assert r.text
        assert r.chunk_id


def test_build_and_retrieve_controlled(tmp_path: Path, monkeypatch) -> None:
    _install_fake_memorag(monkeypatch)

    class FakeEmbedder:
        name = "fake-bge-m3"
        def encode_documents(self, texts):
            return [[0.0] * 8 for _ in texts]
        def encode_query(self, text):
            return [0.0] * 8

    class FakeLLM:
        name = "fake-llm"
        def generate(self, prompt, **kwargs):
            return ""

    docs = _make_corpus()
    cfg = BaselineConfig(name="memorag", mode=RunMode.CONTROLLED, top_k=3)
    baseline = MemoRAGBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
    )
    baseline.build_index(docs)
    results = baseline.retrieve("RAG retrieval", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_retrieve_before_build_raises(tmp_path: Path, monkeypatch) -> None:
    _install_fake_memorag(monkeypatch)
    cfg = BaselineConfig(name="memorag", mode=RunMode.NATIVE, top_k=3)
    baseline = MemoRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    with pytest.raises(RuntimeError, match="build_index"):
        baseline.retrieve("query")


def test_missing_memorag_package_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "memorag", None)
    cfg = BaselineConfig(name="memorag", mode=RunMode.NATIVE, top_k=3)
    baseline = MemoRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    docs = [_make_doc("d1", "text")]
    with pytest.raises(RuntimeError, match="memorag package"):
        baseline.build_index(docs)