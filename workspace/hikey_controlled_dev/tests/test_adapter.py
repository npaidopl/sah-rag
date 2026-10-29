"""Offline tests for the HiKEY baseline adapter."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from baselines.base import BaselineConfig, RunMode
from sah_rag.canonical.models import CanonicalDocument
from workspace.hikey_controlled_dev.adapter import HiKEYBaseline


class MockEmbedder:
    """Deterministic, offline embedder for controlled-mode tests."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return "mock_embedder"

    def _encode(self, text: str) -> list[float]:
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        return [((h[i % len(h)] / 255.0) - 0.5) for i in range(self.dim)]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._encode(text)


def _make_doc(
    doc_id: str, text: str, title: str | None = None
) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset="mock",
        text_raw=text,
        text_normalized=text,
        title=title,
    )


MOCK_CORPUS = [
    _make_doc(
        "d1",
        "Hierarchical retrieval uses a tree of sections and subsections "
        "to find evidence. Tables and figures are first-class units.",
        "Hierarchical Retrieval",
    ),
    _make_doc(
        "d2",
        "Document routing selects top candidates before fine-grained "
        "section scoring. MaxSim ranks units within each section.",
        "Coarse to Fine",
    ),
    _make_doc(
        "d3",
        "Evidence subgraph assembly combines anchor units, governing "
        "headers, and sibling associates.",
        "Evidence Assembly",
    ),
    _make_doc(
        "d4",
        "Open-domain QA requires retrieving passages from large corpora "
        "and reading them with a language model.",
        "ODQA",
    ),
    _make_doc(
        "d5",
        "Multimodal retrieval treats text, tables, and images as separate "
        "modalities with dedicated encoders.",
        "Multimodal",
    ),
]


def test_controlled_mode_build_and_retrieve(tmp_path: Path) -> None:
    work = tmp_path / "hikey_idx"
    cfg = BaselineConfig(name="hikey_controlled", mode=RunMode.CONTROLLED, top_k=3)
    adapter = HiKEYBaseline(
        config=cfg, work_dir=work, embedder=MockEmbedder()
    )
    adapter.build_index(MOCK_CORPUS)
    assert work.exists()
    assert (work / "index_manifest.json").exists()

    results = adapter.retrieve("hierarchical section retrieval", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    doc_ids = {d.doc_id for d in MOCK_CORPUS}
    for r in results:
        assert r.doc_id in doc_ids
        assert r.text
        assert r.score == r.score  # NaN guard
        assert r.metadata.get("approximation") == "hiKEY_text_only_approximation"


def test_retrieve_respects_top_k(tmp_path: Path) -> None:
    cfg = BaselineConfig(name="hikey_controlled", mode=RunMode.CONTROLLED, top_k=2)
    adapter = HiKEYBaseline(
        config=cfg, work_dir=tmp_path / "idx", embedder=MockEmbedder()
    )
    adapter.build_index(MOCK_CORPUS)
    results = adapter.retrieve("evidence assembly", top_k=2)
    assert len(results) <= 2
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_native_mode_is_rejected(tmp_path: Path) -> None:
    cfg = BaselineConfig(name="hikey_controlled", mode=RunMode.NATIVE, top_k=3)
    with pytest.raises(ValueError, match="does not support native mode"):
        HiKEYBaseline(config=cfg, work_dir=tmp_path / "idx")


def test_empty_corpus_returns_empty(tmp_path: Path) -> None:
    cfg = BaselineConfig(name="hikey_controlled", mode=RunMode.CONTROLLED, top_k=3)
    adapter = HiKEYBaseline(
        config=cfg, work_dir=tmp_path / "idx", embedder=MockEmbedder()
    )
    adapter.build_index([])
    assert adapter.retrieve("anything") == []


def test_reported_metrics_attribute_exists() -> None:
    assert hasattr(HiKEYBaseline, "REPORTED_METRICS")
    assert isinstance(HiKEYBaseline.REPORTED_METRICS, dict)


def test_metadata_sections_hierarchy(tmp_path: Path) -> None:
    doc = CanonicalDocument(
        doc_id="h1",
        dataset="mock",
        text_raw="Fallback text that should be ignored when sections exist.",
        text_normalized="Fallback text that should be ignored when sections exist.",
        metadata={
            "sections": [
                {
                    "section_id": "s1",
                    "title": "Introduction",
                    "text": "Introduction body about ODQA.",
                    "level": 1,
                },
                {
                    "section_id": "s2",
                    "title": "Method",
                    "text": "Method body describing hierarchical retrieval.",
                    "level": 1,
                    "parent_id": "s1",
                },
            ]
        },
    )
    cfg = BaselineConfig(name="hikey_controlled", mode=RunMode.CONTROLLED, top_k=5)
    adapter = HiKEYBaseline(
        config=cfg, work_dir=tmp_path / "idx", embedder=MockEmbedder()
    )
    adapter.build_index([doc])
    results = adapter.retrieve("hierarchical retrieval method", top_k=5)
    assert len(results) >= 1
    for r in results:
        assert r.doc_id == "h1"
        assert r.chunk_id in {"s1", "s2"}