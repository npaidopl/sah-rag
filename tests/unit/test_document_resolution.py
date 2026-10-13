"""Tests for document resolution (Step 87)."""

from __future__ import annotations

from rag_system.retrieval.document_resolution import (
    RetrievedDocument,
    resolve_parents_to_documents,
)
from rag_system.retrieval.models import RetrievedParent


def _parent(
    *,
    parent_id: str,
    doc_id: str,
    rank: int,
    score: float,
    chunk_ids: tuple[str, ...] = (),
    dataset: str = "infrastructure-pilot-synthetic-v1",
) -> RetrievedParent:
    ids = chunk_ids or (f"chunk-{parent_id}",)
    return RetrievedParent(
        parent_id=parent_id,
        doc_id=doc_id,
        dataset=dataset,
        rank=rank,
        score=score,
        best_chunk_rank=rank,
        matched_chunk_count=len(ids),
        chunk_ids=ids,
        child_scores=tuple(score for _ in ids),
    )


def test_max_parent_score_per_document():
    parents = (
        _parent(parent_id="p-1", doc_id="doc-A", rank=1, score=0.9),
        _parent(parent_id="p-2", doc_id="doc-A", rank=2, score=0.5),
        _parent(parent_id="p-3", doc_id="doc-B", rank=3, score=0.7),
    )
    docs = resolve_parents_to_documents(parents, query_id="q-1")
    assert docs[0].doc_id == "doc-A"
    assert docs[0].score == 0.9
    assert docs[1].doc_id == "doc-B"


def test_deterministic_tie_break():
    parents = (
        _parent(parent_id="p-1", doc_id="doc-B", rank=1, score=0.5),
        _parent(parent_id="p-2", doc_id="doc-A", rank=2, score=0.5),
    )
    docs = resolve_parents_to_documents(parents, query_id="q-1")
    # Same score, but best_parent_rank differs: p-1 has rank 1
    assert docs[0].doc_id == "doc-B"
    assert docs[1].doc_id == "doc-A"


def test_document_tie_break_by_doc_id_when_rank_equal():
    parents = (
        _parent(parent_id="p-1", doc_id="doc-B", rank=5, score=0.5),
        _parent(parent_id="p-2", doc_id="doc-A", rank=5, score=0.5),
    )
    docs = resolve_parents_to_documents(parents, query_id="q-1")
    assert docs[0].doc_id == "doc-A"


def test_duplicate_content_documents_preserved():
    parents = (
        _parent(parent_id="p-1", doc_id="doc-A", rank=1, score=0.9),
        _parent(parent_id="p-2", doc_id="doc-B", rank=2, score=0.9),
    )
    docs = resolve_parents_to_documents(parents, query_id="q-1")
    assert len(docs) == 2
    assert {d.doc_id for d in docs} == {"doc-A", "doc-B"}


def test_parent_ids_and_chunk_ids_retained():
    parents = (
        _parent(
            parent_id="p-1",
            doc_id="doc-A",
            rank=1,
            score=0.9,
            chunk_ids=("c-1", "c-2"),
        ),
        _parent(
            parent_id="p-2",
            doc_id="doc-A",
            rank=2,
            score=0.5,
            chunk_ids=("c-3",),
        ),
    )
    docs = resolve_parents_to_documents(parents, query_id="q-1")
    assert docs[0].contributing_parent_ids == ("p-1", "p-2")
    assert docs[0].contributing_chunk_ids == ("c-1", "c-2", "c-3")


def test_limit_applied_after_sorting():
    parents = tuple(
        _parent(
            parent_id=f"p-{i}",
            doc_id=f"doc-{i:03d}",
            rank=i,
            score=float(100 - i),
        )
        for i in range(1, 11)
    )
    docs = resolve_parents_to_documents(parents, query_id="q-1", limit=3)
    assert len(docs) == 3
    assert [d.rank for d in docs] == [1, 2, 3]
    assert docs[0].doc_id == "doc-001"


def test_empty_parents_returns_empty():
    assert resolve_parents_to_documents((), query_id="q-1") == ()


def test_retrieved_document_rejects_prohibited_metadata():
    try:
        RetrievedDocument(
            query_id="q-1",
            doc_id="doc-A",
            dataset="ds",
            rank=1,
            score=0.9,
            best_parent_rank=1,
            contributing_parent_ids=("p-1",),
            contributing_chunk_ids=("c-1",),
            metadata={"relevance": 2},
        )
    except Exception as exc:
        assert "prohibited" in str(exc).lower()
    else:
        raise AssertionError("expected validation error")