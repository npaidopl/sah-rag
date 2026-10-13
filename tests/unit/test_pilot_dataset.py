"""Tests for the synthetic pilot dataset builder."""

from __future__ import annotations

from rag_system.pilot_dataset import (
    build_infrastructure_pilot_dataset,
)


def _build():
    return build_infrastructure_pilot_dataset()


def test_exact_dataset_counts():
    docs, chunks, queries, judgments = _build()
    assert len(docs) == 12
    assert len(chunks) == 24
    assert len(queries) == 6
    assert len(judgments) == 12


def test_six_duplicate_content_pairs():
    docs, *_ = _build()
    groups: dict[str, int] = {}
    for d in docs:
        groups[d.content_group_id] = groups.get(d.content_group_id, 0) + 1
    assert len(groups) == 6
    assert all(v == 2 for v in groups.values())


def test_distinct_document_ids_within_pair():
    docs, *_ = _build()
    groups: dict[str, set[str]] = {}
    for d in docs:
        groups.setdefault(d.content_group_id, set()).add(d.doc_id)
    for group, ids in groups.items():
        assert len(ids) == 2, group


def test_same_normalized_text_within_pair():
    docs, *_ = _build()
    groups: dict[str, set[str]] = {}
    for d in docs:
        groups.setdefault(d.content_group_id, set()).add(d.text_normalized)
    for group, texts in groups.items():
        assert len(texts) == 1, group


def test_same_checksum_within_pair():
    docs, *_ = _build()
    groups: dict[str, set[str]] = {}
    for d in docs:
        groups.setdefault(d.content_group_id, set()).add(d.source_text_sha256)
    for group, hashes in groups.items():
        assert len(hashes) == 1, group


def test_two_chunks_per_document():
    docs, chunks, *_ = _build()
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    for d in docs:
        assert per_doc[d.doc_id] == 2


def test_chunk_source_orders_contiguous():
    _, chunks, *_ = _build()
    per_doc: dict[str, list[int]] = {}
    for c in chunks:
        per_doc.setdefault(c.doc_id, []).append(c.source_order)
    for doc_id, orders in per_doc.items():
        assert sorted(orders) == [0, 1], doc_id


def test_document_reconstructs_from_chunks():
    docs, chunks, *_ = _build()
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    doc_by_id = {d.doc_id: d for d in docs}
    for doc_id, cs in by_doc.items():
        cs.sort(key=lambda c: c.source_order)
        rebuilt = " ".join((cs[0].text + " " + cs[1].text).split())
        assert rebuilt == doc_by_id[doc_id].text_normalized


def test_one_query_per_content_group():
    _, _, queries, _ = _build()
    groups = {q.target_content_group_id for q in queries}
    assert len(groups) == 6
    assert len(queries) == 6


def test_two_judgments_per_query():
    _, _, _, judgments = _build()
    per_q: dict[str, int] = {}
    for j in judgments:
        per_q[j.query_id] = per_q.get(j.query_id, 0) + 1
    assert len(per_q) == 6
    assert all(v == 2 for v in per_q.values())


def test_judgment_docs_belong_to_target_group():
    docs, _, queries, judgments = _build()
    doc_group = {d.doc_id: d.content_group_id for d in docs}
    q_group = {q.query_id: q.target_content_group_id for q in queries}
    for j in judgments:
        assert doc_group[j.item_id] == q_group[j.query_id]


def test_no_generated_answer_fields():
    _, _, queries, _ = _build()
    for q in queries:
        dumped = q.model_dump(mode="json")
        assert "answer" not in dumped
        assert "reference_answer" not in dumped
        assert "gold" not in dumped
