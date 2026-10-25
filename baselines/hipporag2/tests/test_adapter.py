"""Offline tests for the HippoRAG2 adapter (controlled mode + native fallback)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from baselines.base import (
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)


# ------------------------------------------------------------
# Mocks
# ------------------------------------------------------------
class MockEmbedder:
    name = "mock-embedder"

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        result = [0.0] * self.dim
        for i, ch in enumerate(text):
            result[i % self.dim] += (ord(ch) % 32) / 32.0
        norm = sum(x * x for x in result) ** 0.5
        if norm > 0.0:
            result = [x / norm for x in result]
        return result


class MockLLM:
    name = "mock-llm"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [
            '{"entities": ["alpha", "beta"], "triples": [["alpha", "related_to", "beta"]]}',
        ]
        self._idx = 0

    def generate(self, prompt: str, **kwargs: object) -> str:
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return resp


# ------------------------------------------------------------
# Corpus helpers
# ------------------------------------------------------------
def _make_doc(doc_id: str, text: str):
    from sah_rag.canonical.models import CanonicalDocument

    return CanonicalDocument(
        doc_id=doc_id,
        dataset="test",
        text_raw=text,
        text_normalized=text,
    )


def _make_corpus():
    return [
        _make_doc("d1", "Alpha is related to beta in the first document."),
        _make_doc("d2", "Gamma and delta are mentioned in the second document."),
        _make_doc("d3", "The epsilon entity appears here."),
        _make_doc("d4", "Zeta connects to eta in this passage."),
        _make_doc("d5", "Theta and iota are final entities."),
    ]


# ------------------------------------------------------------
# Controlled-mode tests
# ------------------------------------------------------------
def test_controlled_build_and_retrieve(tmp_path: Path) -> None:
    try:
        from baselines.hipporag2.adapter import HippoRAG2Baseline
    except ImportError:
        from workspace.hipporag2_dev.adapter import HippoRAG2Baseline

    corpus = _make_corpus()
    cfg = BaselineConfig(name="hipporag2", mode=RunMode.CONTROLLED, top_k=3)
    baseline = HippoRAG2Baseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=MockEmbedder(),
        llm=MockLLM(),
    )
    baseline.build_index(corpus)

    assert (tmp_path / "idx").exists()
    assert (tmp_path / "idx" / "meta.json").exists()
    assert (tmp_path / "idx" / "graph.pkl").exists()
    assert (tmp_path / "idx" / "entity_embeddings.pkl").exists()
    assert (tmp_path / "idx" / "passage_embeddings.pkl").exists()

    results = baseline.retrieve("alpha and beta", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    corpus_ids = {d.doc_id for d in corpus}
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.doc_id in corpus_ids
        assert r.text
        assert isinstance(r.score, float)


def test_controlled_reload_from_disk(tmp_path: Path) -> None:
    try:
        from baselines.hipporag2.adapter import HippoRAG2Baseline
    except ImportError:
        from workspace.hipporag2_dev.adapter import HippoRAG2Baseline

    corpus = _make_corpus()
    cfg = BaselineConfig(name="hipporag2", mode=RunMode.CONTROLLED, top_k=2)
    baseline = HippoRAG2Baseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=MockEmbedder(),
        llm=MockLLM(),
    )
    baseline.build_index(corpus)

    # Fresh instance, same work_dir, no build
    baseline2 = HippoRAG2Baseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=MockEmbedder(),
        llm=MockLLM(),
    )
    results = baseline2.retrieve("alpha", top_k=2)
    assert isinstance(results, list)
    assert all(r.rank == i for i, r in enumerate(results, start=1))


def test_controlled_fallback_when_no_entities(tmp_path: Path) -> None:
    try:
        from baselines.hipporag2.adapter import HippoRAG2Baseline
    except ImportError:
        from workspace.hipporag2_dev.adapter import HippoRAG2Baseline

    corpus = _make_corpus()
    cfg = BaselineConfig(name="hipporag2", mode=RunMode.CONTROLLED, top_k=2)
    baseline = HippoRAG2Baseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=MockEmbedder(),
        llm=MockLLM(responses=['{"entities": []}']),
    )
    baseline.build_index(corpus)
    results = baseline.retrieve("nothing relevant", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_controlled_requires_embedder(tmp_path: Path) -> None:
    try:
        from baselines.hipporag2.adapter import HippoRAG2Baseline
    except ImportError:
        from workspace.hipporag2_dev.adapter import HippoRAG2Baseline

    cfg = BaselineConfig(name="hipporag2", mode=RunMode.CONTROLLED, top_k=2)
    with pytest.raises(ValueError, match="embedder"):
        HippoRAG2Baseline(config=cfg, work_dir=tmp_path / "idx", llm=MockLLM())


def test_controlled_requires_llm(tmp_path: Path) -> None:
    try:
        from baselines.hipporag2.adapter import HippoRAG2Baseline
    except ImportError:
        from workspace.hipporag2_dev.adapter import HippoRAG2Baseline

    cfg = BaselineConfig(name="hipporag2", mode=RunMode.CONTROLLED, top_k=2)
    with pytest.raises(ValueError, match="llm"):
        HippoRAG2Baseline(
            config=cfg, work_dir=tmp_path / "idx", embedder=MockEmbedder()
        )


# ------------------------------------------------------------
# Native-mode tests
# ------------------------------------------------------------
_HIPPORAG_AVAILABLE = importlib.util.find_spec("hipporag") is not None


@pytest.mark.skipif(
    _HIPPORAG_AVAILABLE,
    reason="native-unavailable test only applies when hipporag is NOT installed",
)
def test_native_raises_when_package_missing(tmp_path: Path) -> None:
    try:
        from baselines.hipporag2.adapter import HippoRAG2Baseline
    except ImportError:
        from workspace.hipporag2_dev.adapter import HippoRAG2Baseline

    cfg = BaselineConfig(name="hipporag2", mode=RunMode.NATIVE, top_k=2)
    with pytest.raises(ValueError, match="does not support native"):
        HippoRAG2Baseline(config=cfg, work_dir=tmp_path / "idx")


@pytest.mark.skipif(
    not _HIPPORAG_AVAILABLE,
    reason="hipporag package not installed",
)
def test_native_supports_flag(tmp_path: Path) -> None:
    try:
        from baselines.hipporag2.adapter import HippoRAG2Baseline
    except ImportError:
        from workspace.hipporag2_dev.adapter import HippoRAG2Baseline

    cfg = BaselineConfig(
        name="hipporag2",
        mode=RunMode.NATIVE,
        top_k=2,
        extra={"llm_name": "gpt-4o-mini", "openie_mode": "online"},
    )
    baseline = HippoRAG2Baseline(config=cfg, work_dir=tmp_path / "idx")
    assert baseline.supports_native is True
    assert baseline.config.mode == RunMode.NATIVE