"""Tests for Clerk document-level evaluation mapping."""

from __future__ import annotations

from rag_system.evaluation import (
    ClerkQuery,
    RetrievalJudgment,
    build_clerk_evaluation_case,
    rank_unique_documents,
)
from rag_system.retrieval import RetrievedChunk


def make_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    rank: int,
    score: float,
) -> RetrievedChunk:
    """Create one retrieved child result."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        parent_id=f"parent-{doc_id}",
        doc_id=doc_id,
        dataset="clerk",
        rank=rank,
        score=score,
        text_normalized="Retrieved text.",
        retrieval_method="hybrid_rrf",
        payload={},
    )


def test_rank_unique_documents_preserves_first_rank() -> None:
    """Repeated chunks from one document must collapse."""
    chunks = (
        make_chunk(
            chunk_id="chunk-a1",
            doc_id="doc-a",
            rank=1,
            score=0.9,
        ),
        make_chunk(
            chunk_id="chunk-a2",
            doc_id="doc-a",
            rank=2,
            score=0.8,
        ),
        make_chunk(
            chunk_id="chunk-b1",
            doc_id="doc-b",
            rank=3,
            score=0.7,
        ),
    )

    assert rank_unique_documents(chunks) == (
        "doc-a",
        "doc-b",
    )


def test_document_limit_applies_after_deduplication() -> None:
    """The limit must count unique documents, not chunks."""
    chunks = (
        make_chunk(
            chunk_id="chunk-a1",
            doc_id="doc-a",
            rank=1,
            score=0.9,
        ),
        make_chunk(
            chunk_id="chunk-a2",
            doc_id="doc-a",
            rank=2,
            score=0.8,
        ),
        make_chunk(
            chunk_id="chunk-b1",
            doc_id="doc-b",
            rank=3,
            score=0.7,
        ),
    )

    assert rank_unique_documents(
        chunks,
        limit=1,
    ) == ("doc-a",)


def test_build_clerk_case_uses_document_level() -> None:
    """Clerk cases must evaluate canonical document IDs."""
    query = ClerkQuery(
        query_id="query-001",
        query_text="sample query",
        gold_passage_ids=("doc-b",),
        gold_answer="sample answer",
        split="test",
    )

    judgments = (
        RetrievalJudgment(
            item_id="doc-b",
            relevance=1,
        ),
    )

    chunks = (
        make_chunk(
            chunk_id="chunk-a",
            doc_id="doc-a",
            rank=1,
            score=0.9,
        ),
        make_chunk(
            chunk_id="chunk-b",
            doc_id="doc-b",
            rank=2,
            score=0.8,
        ),
    )

    case = build_clerk_evaluation_case(
        query=query,
        judgments=judgments,
        retrieved_chunks=chunks,
        retrieval_method="hybrid_rrf",
    )

    assert case.evaluation_level == "document"
    assert case.retrieved_ids == (
        "doc-a",
        "doc-b",
    )
    assert case.metadata[
        "gold_passage_count"
    ] == 1

def test_document_chunk_aggregation_rewards_multiple_hits() -> None:
    """Multiple matching chunks may boost one document."""
    from rag_system.evaluation import aggregate_clerk_document_chunks
    chunks = (
        make_chunk(chunk_id="chunk-a1", doc_id="doc-a", rank=1, score=0.60),
        make_chunk(chunk_id="chunk-b1", doc_id="doc-b", rank=2, score=0.58),
        make_chunk(chunk_id="chunk-b2", doc_id="doc-b", rank=3, score=0.40),
    )
    baseline = aggregate_clerk_document_chunks(chunks, additional_chunk_weight=0.0)
    boosted = aggregate_clerk_document_chunks(chunks, additional_chunk_weight=0.10)
    assert baseline[0].doc_id == "doc-a"
    assert boosted[0].doc_id == "doc-b"

def test_document_chunk_aggregation_is_deterministic() -> None:
    """Equivalent inputs should produce equal ordering."""
    from rag_system.evaluation import aggregate_clerk_document_chunks
    chunks = (
        make_chunk(chunk_id="chunk-a", doc_id="doc-a", rank=1, score=0.5),
        make_chunk(chunk_id="chunk-b", doc_id="doc-b", rank=2, score=0.5),
    )
    first = aggregate_clerk_document_chunks(chunks, additional_chunk_weight=0.05)
    second = aggregate_clerk_document_chunks(chunks, additional_chunk_weight=0.05)
    assert first == second
