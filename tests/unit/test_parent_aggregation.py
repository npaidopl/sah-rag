"""Tests for parent-level retrieval aggregation."""

from __future__ import annotations

from rag_system.retrieval import (
    RetrievedChunk,
    aggregate_chunks_to_parents,
)


def make_result(
    *,
    chunk_id: str,
    parent_id: str,
    rank: int,
    score: float,
) -> RetrievedChunk:
    """Create one child retrieval result."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        doc_id=f"doc-{parent_id}",
        dataset="synthetic",
        rank=rank,
        score=score,
        text_normalized="Result text.",
        retrieval_method="hybrid_rrf",
        payload={},
    )


def test_parent_aggregation_groups_children() -> None:
    """Children of the same parent should be grouped."""
    children = (
        make_result(
            chunk_id="chunk-a1",
            parent_id="parent-a",
            rank=1,
            score=0.8,
        ),
        make_result(
            chunk_id="chunk-a2",
            parent_id="parent-a",
            rank=3,
            score=0.4,
        ),
        make_result(
            chunk_id="chunk-b1",
            parent_id="parent-b",
            rank=2,
            score=0.7,
        ),
    )

    parents = aggregate_chunks_to_parents(children)

    assert len(parents) == 2
    assert parents[0].parent_id == "parent-a"
    assert parents[0].matched_chunk_count == 2
    assert parents[0].chunk_ids == (
        "chunk-a1",
        "chunk-a2",
    )


def test_best_child_dominates_parent_score() -> None:
    """Weak extra children should not dominate ranking."""
    children = (
        make_result(
            chunk_id="chunk-a",
            parent_id="parent-a",
            rank=1,
            score=0.9,
        ),
        make_result(
            chunk_id="chunk-b1",
            parent_id="parent-b",
            rank=2,
            score=0.7,
        ),
        make_result(
            chunk_id="chunk-b2",
            parent_id="parent-b",
            rank=3,
            score=0.6,
        ),
    )

    parents = aggregate_chunks_to_parents(
        children,
        additional_child_weight=0.05,
    )

    assert parents[0].parent_id == "parent-a"


def test_parent_limit_is_applied() -> None:
    """Parent result limits should be enforced."""
    children = (
        make_result(
            chunk_id="chunk-a",
            parent_id="parent-a",
            rank=1,
            score=0.9,
        ),
        make_result(
            chunk_id="chunk-b",
            parent_id="parent-b",
            rank=2,
            score=0.8,
        ),
    )

    parents = aggregate_chunks_to_parents(
        children,
        limit=1,
    )

    assert len(parents) == 1
