"""Contract test for the LightRAG baseline (offline, CPU-only)."""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `baselines`, `sah_rag` resolve.
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Ensure workspace is a package (created by setup script, but be safe).
_ws_init = _project_root / "workspace" / "__init__.py"
if not _ws_init.exists() and _ws_init.parent.is_dir():
    _ws_init.write_text("", encoding="utf-8")

import pytest

from baselines.base import BaselineConfig, RunMode, RetrievedChunk
from sah_rag.canonical.models import CanonicalDocument
from baselines.lightrag.adapter import LightRAGBaseline


def _test_config(mode=RunMode.NATIVE, top_k: int = 3) -> BaselineConfig:
    """Test config: strict_full_path=False explicit opt-in (offline fallback)."""
    return BaselineConfig(
        name="lightrag", mode=mode, top_k=top_k,
        extra={"strict_full_path": False, "disable_full_pipeline": True},
    )


class _MockEmbedder:
    name = "mock-embed"

    def encode_documents(self, texts):
        return [[float(len(t) % 7) + 0.1] * 8 for t in texts]

    def encode_query(self, text):
        return [float(len(text) % 7) + 0.1] * 8


class _MockLLM:
    name = "mock-llm"

    def generate(self, prompt, **kwargs):
        return ""


def _make_doc(doc_id: str, text: str) -> CanonicalDocument:
    return CanonicalDocument(
        doc_id=doc_id,
        dataset="mock",
        text_raw=text,
        text_normalized=text,
    )


@pytest.fixture()
def corpus():
    return [
        _make_doc("d1", "The cat sat on the mat in the living room."),
        _make_doc("d2", "Dogs chase balls in the park every morning."),
        _make_doc("d3", "RAG systems retrieve relevant context from documents."),
        _make_doc("d4", "Long-context documents require specialized handling."),
        _make_doc("d5", "Legal contracts are structured hierarchically."),
    ]


def test_build_and_retrieve_native(tmp_path, corpus):
    cfg = _test_config(RunMode.NATIVE, top_k=3)
    baseline = LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    baseline.build_index(corpus)

    assert (tmp_path / "idx").exists()
    assert (tmp_path / "idx" / "chunks.jsonl").exists()
    assert (tmp_path / "idx" / "graph.json").exists()
    assert (tmp_path / "idx" / "index.json").exists()

    for q in ["RAG retrieval", "cat on the mat", "legal contracts"]:
        results = baseline.retrieve(q, top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert len(results) <= 3
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        for r in results:
            assert r.doc_id in {d.doc_id for d in corpus}
            assert r.text


def test_empty_corpus(tmp_path):
    cfg = _test_config(RunMode.NATIVE, top_k=3)
    baseline = LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    baseline.build_index([])
    assert baseline.retrieve("anything", top_k=3) == []


def test_controlled_with_mocks(tmp_path, corpus):
    cfg = _test_config(RunMode.CONTROLLED, top_k=3)
    baseline = LightRAGBaseline(
        config=cfg,
        work_dir=tmp_path / "idx",
        embedder=_MockEmbedder(),
        llm=_MockLLM(),
    )
    baseline.build_index(corpus)
    results = baseline.retrieve("RAG", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_top_k_respected(tmp_path, corpus):
    cfg = _test_config(RunMode.NATIVE, top_k=2)
    baseline = LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    baseline.build_index(corpus)
    results = baseline.retrieve("RAG", top_k=2)
    assert len(results) <= 2


def test_retrieve_before_build(tmp_path):
    cfg = _test_config(RunMode.NATIVE, top_k=3)
    baseline = LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    assert baseline.retrieve("anything", top_k=3) == []


def test_controlled_requires_components(tmp_path):
    cfg = _test_config(RunMode.CONTROLLED, top_k=3)
    with pytest.raises(ValueError):
        LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")


def test_full_pipeline_raises_without_components(tmp_path):
    """Full pipeline raises if components missing (never silent fallback)."""
    cfg = BaselineConfig(
        name="lightrag", mode=RunMode.NATIVE, top_k=3,
        extra={"strict_full_path": False, "disable_full_pipeline": True},
    )
    baseline = LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    docs = [_make_doc("d1", "test doc")]
    with pytest.raises(RuntimeError, match="Vendored|Native mode"):
        baseline._build_index_full(docs)


def test_retrieve_full_raises_without_build(tmp_path):
    """_retrieve_full raises if not built."""
    cfg = BaselineConfig(
        name="lightrag", mode=RunMode.NATIVE, top_k=3,
        extra={"strict_full_path": False, "disable_full_pipeline": True},
    )
    baseline = LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    with pytest.raises(RuntimeError, match="not built"):
        baseline._retrieve_full("query", 3)


def test_strict_full_path_raises_at_build(tmp_path):
    """strict_full_path=true must fail-fast when full pipeline unavailable."""
    cfg = BaselineConfig(
        name="lightrag", mode=RunMode.NATIVE, top_k=3,
        extra={"strict_full_path": True},
    )
    baseline = LightRAGBaseline(config=cfg, work_dir=tmp_path / "idx")
    docs = [_make_doc("d1", "test doc")]
    with pytest.raises(RuntimeError, match="strict_full_path"):
        baseline.build_index(docs)
