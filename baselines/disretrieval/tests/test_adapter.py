"""Offline tests for the DISRetrieval baseline adapter (v0.2.0).

Tests cover:
  - identity (native blocked, controlled active)
  - controlled on-the-fly build with stub components
  - controlled precomputed tree loading (cached embeddings path)
  - contract compliance (contiguous ranks, canonical doc_ids)
  - failure paths (native raises, missing trees raise)

All tests are offline and complete in <10s on CPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from baselines.base import (
    BaselineConfig,
    BaselineStatus,
    RetrievedChunk,
    RunMode,
)
from sah_rag.canonical.models import CanonicalDocument
# Import works both before graduation (workspace/) and after (baselines/)
try:
    from baselines.disretrieval.adapter import DisRetrievalBaseline
except ImportError:
    from workspace.disretrieval_dev.adapter import DisRetrievalBaseline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset="test",
        text_raw=text,
        text_normalized=text,
    )


def _corpus() -> list[CanonicalDocument]:
    return [
        _make_doc("d1", "The cat sat on the mat.\n\nIt was a sunny day."),
        _make_doc("d2", "Dogs chase balls in the park.\n\nThey like to run."),
        _make_doc("d3", "RAG systems retrieve relevant context.\n\nThey help LLMs."),
        _make_doc("d4", "Long-context documents need special handling.\n\nHierarchies help."),
        _make_doc("d5", "Legal contracts are structured hierarchically.\n\nEach clause matters."),
    ]


def _controlled_stub_cfg(top_k: int = 3) -> BaselineConfig:
    return BaselineConfig(
        name="disretrieval",
        mode=RunMode.CONTROLLED,
        top_k=top_k,
        extra={
            "use_stub_components": True,
            "allow_on_the_fly_build": True,
            "native_fidelity": "CONTROLLED_ADAPTATION",
        },
    )


class _MockEmbedder:
    name = "mock_embedder"

    @staticmethod
    def _embed(text: str, dim: int = 8) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [h[i] / 255.0 for i in range(dim)]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._embed(text)


class _MockLLM:
    name = "mock_llm"

    def generate(self, prompt: str, **kwargs) -> str:
        return "summary"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_identity_attributes() -> None:
    assert DisRetrievalBaseline.name == "disretrieval"
    assert DisRetrievalBaseline.version == "0.2.0"
    assert DisRetrievalBaseline.status is BaselineStatus.EXECUTABLE
    assert DisRetrievalBaseline.supports_native is False, (
        "native must remain blocked (missing XLNet backbone)"
    )
    assert DisRetrievalBaseline.supports_controlled is True
    assert DisRetrievalBaseline.is_llm_based is True


# ---------------------------------------------------------------------------
# Native mode must be blocked
# ---------------------------------------------------------------------------

def test_native_mode_raises(tmp_path: Path) -> None:
    """Native mode must raise because supports_native=False."""
    cfg = BaselineConfig(name="disretrieval", mode=RunMode.NATIVE, top_k=3)
    with pytest.raises(ValueError, match="does not support native"):
        DisRetrievalBaseline(config=cfg, work_dir=tmp_path / "idx")


# ---------------------------------------------------------------------------
# Controlled on-the-fly with stub components
# ---------------------------------------------------------------------------

def test_controlled_on_the_fly(tmp_path: Path) -> None:
    docs = _corpus()
    baseline = DisRetrievalBaseline(
        config=_controlled_stub_cfg(top_k=3),
        work_dir=tmp_path / "idx_onfly",
        embedder=_MockEmbedder(),
        llm=_MockLLM(),
    )
    baseline.build_index(docs)

    results = baseline.retrieve("RAG retrieval", top_k=3)
    assert isinstance(results, list)
    assert 0 < len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))

    doc_ids = {d.doc_id for d in docs}
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.doc_id in doc_ids
        assert r.text
        assert isinstance(r.score, float)
        assert r.metadata.get("native_fidelity") == "CONTROLLED_ADAPTATION"


# ---------------------------------------------------------------------------
# Controlled with precomputed trees (cached embeddings path)
# ---------------------------------------------------------------------------

def _write_precomputed_tree(
    root: Path,
    dataset: str,
    doc: CanonicalDocument,
    embedder_name: str,
) -> None:
    """Write a small precomputed tree file for testing."""
    ds_dir = root / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    leaf1 = {
        "id": f"{doc.doc_id}_leaf_0",
        "text": doc.text_normalized,
        "summary": doc.text_normalized,
        "children": [],
        "level": 0,
    }
    leaf2 = {
        "id": f"{doc.doc_id}_leaf_1",
        "text": "second paragraph",
        "summary": "second paragraph",
        "children": [],
        "level": 0,
    }
    root_node = {
        "id": f"{doc.doc_id}_root",
        "text": leaf1["text"] + " " + leaf2["text"],
        "summary": leaf1["summary"] + " " + leaf2["summary"],
        "children": [leaf1, leaf2],
        "level": 1,
    }

    def fake_vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [h[i] / 255.0 for i in range(8)]

    embeddings = {
        leaf1["id"]: fake_vec(leaf1["summary"]),
        leaf2["id"]: fake_vec(leaf2["summary"]),
        root_node["id"]: fake_vec(root_node["summary"]),
    }

    payload = {
        "doc_id": doc.doc_id,
        "tree": root_node,
        "embeddings": embeddings,
        "embedder": {"name": embedder_name, "revision": "test"},
    }
    (ds_dir / f"doc_{doc.doc_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_controlled_precomputed_trees(tmp_path: Path) -> None:
    """Load trees from disk; cached embeddings match injected embedder."""
    docs = _corpus()
    precomputed_root = tmp_path / "trees"
    for doc in docs:
        _write_precomputed_tree(precomputed_root, "test", doc, "mock_embedder")

    cfg = BaselineConfig(
        name="disretrieval",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={
            "precomputed_trees_root": str(precomputed_root),
            "build_from_precomputed": True,
            "allow_on_the_fly_build": False,
            "use_stub_components": False,
            "dataset_id_map": {"test": "test"},
            "native_fidelity": "CONTROLLED_ADAPTATION",
        },
    )

    baseline = DisRetrievalBaseline(
        config=cfg,
        work_dir=tmp_path / "idx_precomputed",
        embedder=_MockEmbedder(),
        llm=_MockLLM(),
    )
    baseline.build_index(docs)

    results = baseline.retrieve("RAG retrieval", top_k=3)
    assert 0 < len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    doc_ids = {d.doc_id for d in docs}
    for r in results:
        assert r.doc_id in doc_ids


def test_precomputed_missing_and_onfly_disabled_raises(tmp_path: Path) -> None:
    """No precomputed trees and allow_on_the_fly_build=False must raise."""
    docs = _corpus()
    cfg = BaselineConfig(
        name="disretrieval",
        mode=RunMode.CONTROLLED,
        top_k=3,
        extra={
            "precomputed_trees_root": str(tmp_path / "nonexistent_trees"),
            "build_from_precomputed": True,
            "allow_on_the_fly_build": False,
            "use_stub_components": False,
        },
    )
    baseline = DisRetrievalBaseline(
        config=cfg,
        work_dir=tmp_path / "idx_missing",
        embedder=_MockEmbedder(),
        llm=_MockLLM(),
    )
    with pytest.raises(RuntimeError, match="No precomputed trees"):
        baseline.build_index(docs)


# ---------------------------------------------------------------------------
# Contract compliance
# ---------------------------------------------------------------------------

def test_rank_contiguity(tmp_path: Path) -> None:
    docs = _corpus()
    baseline = DisRetrievalBaseline(
        config=_controlled_stub_cfg(top_k=5),
        work_dir=tmp_path / "idx_ranks",
        embedder=_MockEmbedder(),
        llm=_MockLLM(),
    )
    baseline.build_index(docs)
    results = baseline.retrieve("cat", top_k=5)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_empty_retrieve_before_build(tmp_path: Path) -> None:
    baseline = DisRetrievalBaseline(
        config=_controlled_stub_cfg(top_k=3),
        work_dir=tmp_path / "idx_empty",
        embedder=_MockEmbedder(),
        llm=_MockLLM(),
    )
    assert baseline.retrieve("anything", top_k=3) == []


def test_manifest_written(tmp_path: Path) -> None:
    docs = _corpus()
    baseline = DisRetrievalBaseline(
        config=_controlled_stub_cfg(top_k=3),
        work_dir=tmp_path / "idx_manifest",
        embedder=_MockEmbedder(),
        llm=_MockLLM(),
    )
    baseline.build_index(docs)

    manifest_path = tmp_path / "idx_manifest" / "trees_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["document_count"] == len(docs)
    assert set(payload["documents"]) == {d.doc_id for d in docs}
    assert payload["mode"] == "controlled"
    assert payload["native_fidelity"] == "CONTROLLED_ADAPTATION"


def test_native_raises_clear_error(tmp_path: Path) -> None:
    """Native mode raises an actionable error when backbone is missing."""
    import os
    # Ensure env doesn't accidentally point to a real backbone
    old = os.environ.get("RAG_NATIVE_BACKBONE_ROOT")
    os.environ["RAG_NATIVE_BACKBONE_ROOT"] = str(tmp_path / "nonexistent_backbone")
    try:
        cfg = BaselineConfig(name="disretrieval", mode=RunMode.NATIVE, top_k=3)
        # Skip _validate (which rejects native since supports_native=False)
        # by constructing with the executable contract temporarily bypassed
        baseline = object.__new__(DisRetrievalBaseline)
        baseline.config = cfg
        baseline.work_dir = tmp_path / "idx"
        baseline.work_dir.mkdir(parents=True, exist_ok=True)
        baseline.embedder = None
        baseline.llm = None
        baseline._trees = {}
        baseline._embedding_cache = {}
        baseline._parser = None
        baseline._summarizer = None
        baseline._embedder_backend = None
        with pytest.raises(RuntimeError, match="NATIVE_BACKBONE_MISSING"):
            baseline._ensure_native_components()
    finally:
        if old is None:
            os.environ.pop("RAG_NATIVE_BACKBONE_ROOT", None)
        else:
            os.environ["RAG_NATIVE_BACKBONE_ROOT"] = old
