"""Tests for Multi-Meta-RAG adapter (native + controlled).

All tests are offline, use mocks, and complete in < 10s on CPU.
"""
from __future__ import annotations

import pytest

from baselines.base import BaselineConfig, RetrievedChunk, RunMode
from sah_rag.canonical.models import CanonicalDocument
from baselines.multi_meta_rag.adapter import MultiMetaRAGBaseline


class MockEmbedder:
    name = "mock-embedder"
    dim = 128
    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for w in text.lower().split():
            h = 0
            for c in w:
                h = (h * 31 + ord(c)) & 0xFFFFFFFF
            vec[h % self.dim] += 1.0
        n = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / n for x in vec]
    def encode_documents(self, ts): return [self._encode(t) for t in ts]
    def encode_query(self, t): return self._encode(t)


class MockLLM:
    """Deterministic LLM mock: substring-of-prompt -> JSON response."""
    name = "mock-llm"
    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0
    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        for key, val in self.responses.items():
            if key.lower() in prompt.lower():
                return val
        return '{"article source": null, "publication date": null}'


def make_doc(doc_id, text, source=None):
    md = {}
    if source is not None: md["source"] = source
    return CanonicalDocument(
        doc_id=doc_id, dataset="test", text_raw=text,
        text_normalized=text, metadata=md,
    )


# ===== Controlled (LLM-free) =====

def _build_controlled(tmp_path, docs, top_k=3):
    cfg = BaselineConfig(name="multi_meta_rag", mode=RunMode.CONTROLLED, top_k=top_k)
    b = MultiMetaRAGBaseline(config=cfg, work_dir=tmp_path / "idx",
                              embedder=MockEmbedder(), llm=None)
    b.build_index(docs)
    return b


def test_controlled_build_creates_index_files(tmp_path):
    docs = [make_doc("d1", "alpha beta", source="SourceA"),
            make_doc("d2", "gamma delta", source="SourceB")]
    _build_controlled(tmp_path, docs)
    assert (tmp_path / "idx").exists()
    assert any((tmp_path / "idx").iterdir())


def test_controlled_retrieve_contiguous_ranks(tmp_path):
    docs = [make_doc("d1", "alpha beta gamma", source="SourceA"),
            make_doc("d2", "beta gamma delta", source="SourceB"),
            make_doc("d3", "gamma delta epsilon", source="SourceA")]
    b = _build_controlled(tmp_path, docs, top_k=3)
    r = b.retrieve("alpha beta", top_k=3)
    assert isinstance(r, list)
    assert all(isinstance(x, RetrievedChunk) for x in r)
    assert [x.rank for x in r] == list(range(1, len(r) + 1))


def test_controlled_respects_top_k(tmp_path):
    docs = [make_doc(f"d{i}", f"document {i}", source="SourceA") for i in range(5)]
    b = _build_controlled(tmp_path, docs, top_k=10)
    assert len(b.retrieve("document", top_k=2)) <= 2


def test_controlled_filter_by_source(tmp_path):
    docs = [make_doc("d1", "The Verge covers tech", source="The Verge"),
            make_doc("d2", "TechCrunch covers startups", source="TechCrunch"),
            make_doc("d3", "The Verge reviews phones", source="The Verge")]
    b = _build_controlled(tmp_path, docs, top_k=10)
    r = b.retrieve("news from The Verge about gadgets", top_k=10)
    assert len(r) == 2
    assert {x.doc_id for x in r} == {"d1", "d3"}


def test_controlled_no_filter_when_no_source_in_query(tmp_path):
    docs = [make_doc("d1", "alpha text", source="SourceA"),
            make_doc("d2", "beta text", source="SourceB")]
    b = _build_controlled(tmp_path, docs, top_k=10)
    assert len(b.retrieve("something unrelated", top_k=10)) == 2


def test_controlled_missing_source_metadata(tmp_path):
    docs = [make_doc("d1", "alpha"), make_doc("d2", "beta")]
    b = _build_controlled(tmp_path, docs, top_k=10)
    assert len(b.retrieve("anything", top_k=10)) == 2


# ===== Native (LLM-based) =====

def _build_native(tmp_path, docs, top_k=3, llm_responses=None):
    cfg = BaselineConfig(name="multi_meta_rag", mode=RunMode.NATIVE, top_k=top_k)
    b = MultiMetaRAGBaseline(config=cfg, work_dir=tmp_path / "idx",
                              embedder=MockEmbedder(),
                              llm=MockLLM(llm_responses or {}))
    b.build_index(docs)
    return b


def test_native_requires_prereqs(tmp_path):
    cfg = BaselineConfig(
        name="multi_meta_rag", mode=RunMode.NATIVE, top_k=3,
        extra={"native_embedder_path": str(tmp_path / "missing_embedder")},
    )
    with pytest.raises((FileNotFoundError, EnvironmentError)):
        MultiMetaRAGBaseline(config=cfg, work_dir=tmp_path / "idx")


def test_native_llm_extraction_filters(tmp_path):
    docs = [make_doc("d1", "The Verge covers tech", source="The Verge"),
            make_doc("d2", "TechCrunch covers startups", source="TechCrunch")]
    llm_resp = {"The Verge": '{"article source": "The Verge", "publication date": null}'}
    b = _build_native(tmp_path, docs, top_k=10, llm_responses=llm_resp)
    r = b.retrieve("what did The Verge say", top_k=10)
    assert len(r) == 1
    assert r[0].doc_id == "d1"
    assert b.llm.call_count > 0


def test_native_null_source_returns_all(tmp_path):
    docs = [make_doc("d1", "alpha", source="SourceA"),
            make_doc("d2", "beta", source="SourceB")]
    llm_resp = {"general": '{"article source": null, "publication date": null}'}
    b = _build_native(tmp_path, docs, top_k=10, llm_responses=llm_resp)
    assert len(b.retrieve("general question", top_k=10)) == 2


def test_native_chunking_produces_multiple_chunks(tmp_path):
    long_text = " ".join(f"word{i}" for i in range(300))
    docs = [make_doc("d1", long_text, source="SourceA")]
    llm_resp = {"anything": '{"article source": "SourceA", "publication date": null}'}
    b = _build_native(tmp_path, docs, top_k=100, llm_responses=llm_resp)
    r = b.retrieve("anything", top_k=100)
    chunk_ids = {c.chunk_id for c in r}
    assert len(chunk_ids) > 1
    assert all(c.doc_id == "d1" for c in r)


def test_native_and_controlled_same_interface(tmp_path):
    docs = [make_doc("d1", "alpha beta", source="SourceA")]
    b_native = _build_native(tmp_path / "n", docs, top_k=3,
                              llm_responses={"q": '{"article source": "SourceA"}'})
    b_ctrl = _build_controlled(tmp_path / "c", docs, top_k=3)
    for b in (b_native, b_ctrl):
        r = b.retrieve("q", top_k=3)
        assert isinstance(r, list)
        assert all(isinstance(x, RetrievedChunk) for x in r)
        if r:
            assert [x.rank for x in r] == list(range(1, len(r) + 1))