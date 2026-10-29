"""Offline tests for the HiKEY native pipeline (mock mode)."""
from __future__ import annotations

import json
from pathlib import Path

from workspace.hikey_native_dev.components.card_builder import build_section_tree
from workspace.hikey_native_dev.components.dhp_parser import parse_pdf
from workspace.hikey_native_dev.components.evidence_assembler import (
    Unit,
    assemble_evidence,
)
from workspace.hikey_native_dev.components.maxsim import cosine, maxsim_section_score
from workspace.hikey_native_dev.components.multimodal_encoder import MultiModalEncoder
from workspace.hikey_native_dev.pipeline import HiKEYNativePipeline


def test_mock_parse_returns_tree() -> None:
    blocks = parse_pdf(Path("dummy.pdf"), mock=True)
    assert len(blocks) > 0
    assert any(b.block_type == "title" for b in blocks)
    assert any(b.block_type == "header" for b in blocks)
    assert any(b.block_type == "text" for b in blocks)


def test_build_section_tree() -> None:
    blocks = parse_pdf(Path("dummy.pdf"), mock=True)
    doc = build_section_tree(blocks, doc_id="dummy")
    assert doc.doc_id == "dummy"
    assert len(doc.sections) >= 3
    assert all(s.units for s in doc.sections)


def test_maxsim() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.9, 0.1, 0.0]
    c = [0.0, 1.0, 0.0]
    assert abs(cosine(a, b) - 0.9938) < 1e-3
    assert maxsim_section_score(a, [c, b]) > maxsim_section_score(a, [c])


def test_evidence_assembly_three_phases() -> None:
    anchor = Unit("u1", "d1", "s1", ["T", "S1"], "text", "content", score=0.9)
    sib_table = Unit("u2", "d1", "s1", ["T", "S1"], "table", "table content", score=0.5)
    sem_fig = Unit("u3", "d1", "s2", ["T", "S2"], "figure", "figure content", score=0.7)
    sections = [("s1", [anchor])]
    all_units = {"s1": [anchor, sib_table], "s2": [sem_fig]}
    sub = assemble_evidence(
        sections,
        all_units_by_section=all_units,
        similarity_lookup=lambda a, b: 0.9 if a.unit_type != b.unit_type else 0.1,
        token_budget=1000,
    )
    ids = {u.unit_id for u in sub.units}
    assert "u1" in ids
    assert "u2" in ids  # sibling packed
    assert "u3" in ids  # semantic associate packed


def test_encoder_mock_deterministic() -> None:
    enc = MultiModalEncoder(mock=True)
    a = enc.encode_texts(["hello world"])[0]
    b = enc.encode_texts(["hello world"])[0]
    assert a == b
    c = enc.encode_texts(["different"])[0]
    assert a != c


def test_pipeline_end_to_end_mock(tmp_path: Path) -> None:
    """Full pipeline smoke test with mock components."""
    # Fake PDFs (mock parser ignores actual content)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.pdf").write_bytes(b"fake pdf 1")
    (corpus / "doc2.pdf").write_bytes(b"fake pdf 2")

    pipeline = HiKEYNativePipeline(work_dir=tmp_path / "work", mock=True)
    pipeline.index(corpus)
    assert len(pipeline.docs) == 2
    assert (tmp_path / "work" / "manifest.json").exists()

    results = pipeline.retrieve("introduction method", top_k=5)
    assert isinstance(results, list)
    assert len(results) >= 1
    ranks = [r["rank"] for r in results]
    assert ranks == list(range(1, len(results) + 1))
    for r in results:
        assert r["doc_id"] in {"doc1", "doc2"}
        assert r["chunk_id"]
        assert r["metadata"]["source"] == "hikey_native_paper_exact"

    answer = pipeline.answer("introduction", top_k=3)
    assert answer["answer"].startswith("[MOCK_ANSWER]")


def test_pipeline_empty_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "empty"
    corpus.mkdir()
    pipeline = HiKEYNativePipeline(work_dir=tmp_path / "work", mock=True)
    # mock=True means no PDFs is OK
    pipeline.index(corpus)
    assert pipeline.retrieve("anything") == []