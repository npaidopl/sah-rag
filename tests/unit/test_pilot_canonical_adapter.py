"""Tests for the pilot canonical adapter logic."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sah_rag.canonical.pilot_adapter import (
    adapt_pilot_dataset_to_canonical,
)

SOURCE = Path("data/pilot/infrastructure-pilot-synthetic-v1")


def _adapt():
    return adapt_pilot_dataset_to_canonical(pilot_root=SOURCE)


def test_exact_document_count():
    documents, _, _, _ = _adapt()
    assert len(documents) == 12


def test_exact_example_count():
    _, examples, _, _ = _adapt()
    assert len(examples) == 6


def test_document_ids_preserved():
    documents, _, _, _ = _adapt()
    ids = {d.doc_id for d in documents}
    assert ids == {f"pilot-doc-{i:02d}" for i in range(1, 13)}


def test_query_ids_preserved():
    _, examples, _, _ = _adapt()
    qids = {e.query.query_id for e in examples}
    assert qids == {f"pilot-query-{i:02d}" for i in range(1, 7)}


def test_duplicate_content_pairs_preserved():
    documents, _, validation, _ = _adapt()
    assert validation.duplicate_content_preserved is True
    groups: dict[str, set[str]] = {}
    for d in documents:
        gid = str(d.metadata["source_content_group_id"])
        groups.setdefault(gid, set()).add(d.text_normalized)
    assert len(groups) == 6
    for gid, texts in groups.items():
        assert len(texts) == 1, gid


def test_two_relevant_documents_per_example():
    _, examples, _, _ = _adapt()
    for e in examples:
        assert len(e.ground_truth.relevant_ids) == 2


def test_judgment_to_document_mapping_valid():
    _, examples, validation, _ = _adapt()
    assert validation.relevant_document_mapping_valid is True


def test_source_chunk_reconstruction_validated():
    _, _, validation, _ = _adapt()
    assert validation.source_chunk_reconstruction_valid is True


def test_source_chunks_not_authoritative():
    _, _, _, mapping = _adapt()
    policy = mapping["source_chunk_policy"]
    assert policy["consumed_as_authoritative_chunks"] is False
    assert policy["role"] == "validation_reference_only"


def test_no_answer_fields_created():
    _, examples, validation, _ = _adapt()
    assert validation.answer_fields_absent is True
    for e in examples:
        dumped = e.model_dump(mode="json")
        assert "answer" not in dumped
        assert "reference_answer" not in dumped
        assert e.ground_truth.answers == []


def test_source_artifacts_remain_unchanged():
    def tree_sha(root: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(root.iterdir()):
            if p.is_file():
                h.update(p.name.encode("utf-8"))
                h.update(p.read_bytes())
        return h.hexdigest()

    before = tree_sha(SOURCE)
    _adapt()
    after = tree_sha(SOURCE)
    assert before == after
