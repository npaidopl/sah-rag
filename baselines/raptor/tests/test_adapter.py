"""Offline contract-compliance tests for the RAPTOR baseline adapter."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from baselines.base import BaselineConfig, RunMode
from sah_rag.canonical.models import CanonicalDocument
from baselines.raptor.adapter import RaptorBaseline


class _MockEmbedder:
    """Deterministic embedder; matches SAH-RAG Embedder protocol."""

    name = "mock-embedder"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(42)
        return [rng.standard_normal(self.dim).tolist() for _ in texts]

    def encode_query(self, text: str) -> list[float]:
        rng = np.random.default_rng(123)
        return rng.standard_normal(self.dim).tolist()


class _MockLLM:
    """Deterministic summarizer; matches SAH-RAG LLM protocol."""

    name = "mock-llm"

    def generate(self, prompt: str, **kwargs) -> str:
        return "MOCK SUMMARY: " + prompt[:40].replace("\n", " ")


def _make_doc(doc_id: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset="test",
        text_raw=text,
        text_normalized=text,
    )


def test_controlled_build_and_retrieve(tmp_path: Path) -> None:
    docs = [
        _make_doc(f"d{i}", f"Document {i} about topic {i}: "
                            f"content about RAG, retrieval, and evidence.")
        for i in range(1, 11)
    ]
    cfg = BaselineConfig(
        name="raptor",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={"chunk_size": 20, "max_levels": 2, "cluster_size": 2},
    )
    baseline = RaptorBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=_MockEmbedder(dim=8),
        llm=_MockLLM(),
    )
    baseline.build_index(docs)
    assert (tmp_path / "idx" / "raptor_tree.jsonl").exists()
    assert (tmp_path / "idx" / "raptor_embeddings.npy").exists()

    results = baseline.retrieve("RAG retrieval", top_k=3)
    assert isinstance(results, list)
    assert 0 < len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    corpus_ids = {d.doc_id for d in docs}
    for r in results:
        assert r.doc_id in corpus_ids
        assert r.text
        assert not math.isnan(r.score)
        assert r.chunk_id


def test_controlled_retrieve_top_k_defaults(tmp_path: Path) -> None:
    docs = [_make_doc(f"d{i}", f"Doc {i}") for i in range(1, 6)]
    cfg = BaselineConfig(
        name="raptor", mode=RunMode.CONTROLLED, top_k=2,
        extra={"chunk_size": 10, "max_levels": 1, "cluster_size": 2},
    )
    baseline = RaptorBaseline(
        config=cfg,
        work_dir=tmp_path / "idx2",
        embedder=_MockEmbedder(dim=4),
        llm=_MockLLM(),
    )
    baseline.build_index(docs)
    results = baseline.retrieve("query")
    assert len(results) <= 2


def test_native_requires_paths(tmp_path: Path) -> None:
    cfg = BaselineConfig(
        name="raptor", mode=RunMode.NATIVE, top_k=3, extra={},
    )
    with pytest.raises(ValueError, match="embedder_path"):
        RaptorBaseline(config=cfg, work_dir=tmp_path / "idx3")


def test_controlled_requires_embedder(tmp_path: Path) -> None:
    """The frozen base contract rejects controlled mode without an embedder."""
    cfg = BaselineConfig(
        name="raptor", mode=RunMode.CONTROLLED, top_k=3, extra={},
    )
    with pytest.raises(ValueError, match="requires embedder injection"):
        RaptorBaseline(config=cfg, work_dir=tmp_path / "idx4")


def test_controlled_requires_llm_when_llm_based(tmp_path: Path) -> None:
    """LLM-based baselines must receive an injected llm in controlled mode."""
    cfg = BaselineConfig(
        name="raptor", mode=RunMode.CONTROLLED, top_k=3, extra={},
    )
    with pytest.raises(ValueError, match="requires llm injection"):
        RaptorBaseline(
            config=cfg,
            work_dir=tmp_path / "idx5",
            embedder=_MockEmbedder(dim=4),
        )


@pytest.mark.skipif(True, reason="requires real models + GPU; runs on training server")
def test_native_end_to_end(tmp_path: Path) -> None:
    """Placeholder: enable on the GPU server with real embedder_path / llm_path."""
    pass