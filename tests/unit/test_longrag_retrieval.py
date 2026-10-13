"""Tests for LongRAG dense retrieval and conversion."""

from __future__ import annotations

from rag_system.baselines import (
    RetrievedLongUnit,
    retrieved_to_ranked_long_units,
)


def make_retrieved(
    *,
    long_unit_id: str = "u1",
    rank: int = 1,
    score: float = 0.9,
) -> RetrievedLongUnit:
    return RetrievedLongUnit(
        long_unit_id=long_unit_id,
        condition_id="LONGRAG-SD-4096",
        dataset_id="clerk",
        rank=rank,
        score=score,
        construction_policy="single_document_if_within_budget",
        target_token_size=4096,
        rendered_token_count=100,
        member_doc_ids=("doc-a",),
        member_parent_ids=("parent-a",),
        member_chunk_ids=("chunk-a",),
        payload={"long_unit_id": long_unit_id},
    )


def test_retrieved_to_ranked_preserves_rank_score_identity() -> None:
    retrieved = (make_retrieved(rank=1, score=0.93),)
    ranked = retrieved_to_ranked_long_units(
        query_id="q1",
        retrieved=retrieved,
    )
    assert len(ranked) == 1
    assert ranked[0].rank == 1
    assert ranked[0].score == 0.93
    assert ranked[0].long_unit_id == "u1"
    assert ranked[0].member_doc_ids == ("doc-a",)
    assert ranked[0].query_id == "q1"


def test_retrieved_to_ranked_preserves_order() -> None:
    retrieved = (
        make_retrieved(long_unit_id="u1", rank=1, score=0.95),
        make_retrieved(long_unit_id="u2", rank=2, score=0.85),
        make_retrieved(long_unit_id="u3", rank=3, score=0.75),
    )
    ranked = retrieved_to_ranked_long_units(
        query_id="q1",
        retrieved=retrieved,
    )
    assert [r.long_unit_id for r in ranked] == ["u1", "u2", "u3"]
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_retrieved_multi_document_lineage_is_native() -> None:
    retrieved = (
        RetrievedLongUnit(
            long_unit_id="u1",
            condition_id="LONGRAG-ADJ-4096",
            dataset_id="clerk",
            rank=1,
            score=0.9,
            construction_policy="adjacent_document_segments",
            target_token_size=4096,
            rendered_token_count=4000,
            member_doc_ids=("doc-a", "doc-b", "doc-c"),
            member_parent_ids=("parent-a", "parent-b"),
            member_chunk_ids=("chunk-a", "chunk-b"),
            payload={},
        ),
    )
    ranked = retrieved_to_ranked_long_units(
        query_id="q1",
        retrieved=retrieved,
    )
    assert ranked[0].member_doc_ids == ("doc-a", "doc-b", "doc-c")
    assert ranked[0].member_parent_ids == ("parent-a", "parent-b")
