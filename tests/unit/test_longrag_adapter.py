"""Tests for deterministic LongRAG adapter artifacts."""

from __future__ import annotations

import pytest

from baselines import (
    LongRAGSourceSegment,
    RankedLongUnit,
    construct_long_units,
    expand_ranked_long_units_to_documents,
    summarize_long_units,
)


def make_segment(
    doc_id: str,
    parent_id: str,
    chunk_id: str,
    document_order: int,
    parent_order: int,
    chunk_order: int,
    token_count: int,
) -> LongRAGSourceSegment:
    """Create one test source segment."""
    return LongRAGSourceSegment(
        dataset_id="synthetic",
        doc_id=doc_id,
        parent_id=parent_id,
        chunk_id=chunk_id,
        document_order=document_order,
        parent_order=parent_order,
        chunk_order=chunk_order,
        text=f"Text for {chunk_id}.",
        token_count=token_count,
    )


def test_adjacent_construction_is_deterministic() -> None:
    """Equivalent inputs must create equal long units."""
    segments = (
        make_segment(
            doc_id="doc-a",
            parent_id="parent-a",
            chunk_id="chunk-a1",
            document_order=0,
            parent_order=0,
            chunk_order=0,
            token_count=4,
        ),
        make_segment(
            doc_id="doc-b",
            parent_id="parent-b",
            chunk_id="chunk-b1",
            document_order=1,
            parent_order=0,
            chunk_order=0,
            token_count=4,
        ),
        make_segment(
            doc_id="doc-c",
            parent_id="parent-c",
            chunk_id="chunk-c1",
            document_order=2,
            parent_order=0,
            chunk_order=0,
            token_count=4,
        ),
    )

    first = construct_long_units(
        dataset_id="synthetic",
        segments=segments,
        target_token_size=8,
        construction_policy="adjacent_segments",
    )

    second = construct_long_units(
        dataset_id="synthetic",
        segments=tuple(reversed(segments)),
        target_token_size=8,
        construction_policy="adjacent_segments",
    )

    assert first == second
    assert len(first) == 2

    assert first[0].member_doc_ids == (
        "doc-a",
        "doc-b",
    )

    assert first[1].member_doc_ids == (
        "doc-c",
    )


def test_parent_grouping_does_not_cross_parents() -> None:
    """Parent-grouped units must preserve boundaries."""
    segments = (
        make_segment(
            doc_id="doc-a",
            parent_id="parent-a",
            chunk_id="chunk-a1",
            document_order=0,
            parent_order=0,
            chunk_order=0,
            token_count=3,
        ),
        make_segment(
            doc_id="doc-a",
            parent_id="parent-a",
            chunk_id="chunk-a2",
            document_order=0,
            parent_order=0,
            chunk_order=1,
            token_count=3,
        ),
        make_segment(
            doc_id="doc-a",
            parent_id="parent-b",
            chunk_id="chunk-b1",
            document_order=0,
            parent_order=1,
            chunk_order=0,
            token_count=3,
        ),
    )

    units = construct_long_units(
        dataset_id="synthetic",
        segments=segments,
        target_token_size=10,
        construction_policy="structure_preserving_parent_groups",
    )

    assert len(units) == 2

    assert units[0].member_chunk_ids == (
        "chunk-a1",
        "chunk-a2",
    )

    assert units[1].member_chunk_ids == (
        "chunk-b1",
    )


def test_summary_requires_complete_coverage() -> None:
    """The summary must report exact source coverage."""
    segments = (
        make_segment(
            doc_id="doc-a",
            parent_id="parent-a",
            chunk_id="chunk-a1",
            document_order=0,
            parent_order=0,
            chunk_order=0,
            token_count=4,
        ),
        make_segment(
            doc_id="doc-b",
            parent_id="parent-b",
            chunk_id="chunk-b1",
            document_order=1,
            parent_order=0,
            chunk_order=0,
            token_count=4,
        ),
    )

    units = construct_long_units(
        dataset_id="synthetic",
        segments=segments,
        target_token_size=8,
        construction_policy="adjacent_segments",
    )

    summary = summarize_long_units(
        dataset_id="synthetic",
        construction_policy="adjacent_segments",
        target_token_size=8,
        source_segments=segments,
        long_units=units,
    )

    assert summary.long_unit_count == 1
    assert summary.complete_document_coverage is True
    assert summary.complete_parent_coverage is True
    assert summary.complete_chunk_coverage is True


def test_ranked_units_expand_documents_deterministically() -> None:
    """Document expansion must deduplicate by first occurrence."""
    ranked = (
        RankedLongUnit(
            query_id="query-001",
            rank=1,
            long_unit_id="unit-1",
            score=0.9,
            construction_policy="adjacent_segments",
            member_doc_ids=(
                "doc-a",
                "doc-b",
            ),
            member_parent_ids=(
                "parent-a",
                "parent-b",
            ),
            member_chunk_ids=(
                "chunk-a",
                "chunk-b",
            ),
        ),
        RankedLongUnit(
            query_id="query-001",
            rank=2,
            long_unit_id="unit-2",
            score=0.8,
            construction_policy="adjacent_segments",
            member_doc_ids=(
                "doc-b",
                "doc-c",
            ),
            member_parent_ids=(
                "parent-b",
                "parent-c",
            ),
            member_chunk_ids=(
                "chunk-b",
                "chunk-c",
            ),
        ),
    )

    expanded = expand_ranked_long_units_to_documents(
        query_id="query-001",
        ranked_long_units=ranked,
        document_limit=10,
    )

    assert tuple(
        result.doc_id
        for result in expanded
    ) == (
        "doc-a",
        "doc-b",
        "doc-c",
    )

    assert expanded[2].source_long_unit_id == (
        "unit-2"
    )

    assert expanded[2].membership_position == 1


def test_duplicate_chunk_ids_are_rejected() -> None:
    """Source chunk identities must be globally unique."""
    segment = make_segment(
        doc_id="doc-a",
        parent_id="parent-a",
        chunk_id="chunk-a",
        document_order=0,
        parent_order=0,
        chunk_order=0,
        token_count=4,
    )

    with pytest.raises(
        ValueError,
        match="duplicate chunk IDs",
    ):
        construct_long_units(
            dataset_id="synthetic",
            segments=(
                segment,
                segment,
            ),
            target_token_size=8,
            construction_policy="adjacent_segments",
        )


def test_oversized_source_segment_is_rejected() -> None:
    """Oversized source segments require upstream splitting."""
    segment = make_segment(
        doc_id="doc-a",
        parent_id="parent-a",
        chunk_id="chunk-a",
        document_order=0,
        parent_order=0,
        chunk_order=0,
        token_count=20,
    )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        construct_long_units(
            dataset_id="synthetic",
            segments=(segment,),
            target_token_size=10,
            construction_policy="adjacent_segments",
        )
