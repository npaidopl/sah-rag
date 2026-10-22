"""Tests for parser profiling."""

from __future__ import annotations

from sah_rag.artifacts import (
    BoundaryEvidence,
    BoundaryMethod,
    OffsetSpan,
    ParsedDocument,
    SemanticParent,
)
from sah_rag.parsing import (
    ParserProfileAccumulator,
)


def make_parsed_document() -> ParsedDocument:
    """Create a small parsed document for profiling."""

    parent = SemanticParent(
        parent_id="parent-001",
        doc_id="doc-001",
        dataset="synthetic",
        parent_parent_id=None,
        hierarchy_level=0,
        sibling_index=0,
        title="CHAPTER 1",
        text_normalized="CHAPTER 1\n\nBody.",
        text_raw="CHAPTER 1\n\nBody.",
        normalized_char_span=OffsetSpan(
            start=0,
            end=16,
        ),
        raw_char_span=OffsetSpan(
            start=0,
            end=16,
        ),
        token_span=None,
        token_count=None,
        start_boundary=(
            BoundaryEvidence(
                method=BoundaryMethod.DOCUMENT_START,
                confidence=1.0,
            ),
        ),
        end_boundary=(
            BoundaryEvidence(
                method=BoundaryMethod.DOCUMENT_END,
                confidence=1.0,
            ),
        ),
        parser_id="parser",
        parser_version="0.1.0",
        parser_config_hash="hash",
        metadata={
            "heading_detected": True,
        },
    )

    return ParsedDocument(
        doc_id="doc-001",
        dataset="synthetic",
        parser_id="parser",
        parser_version="0.1.0",
        parser_config_hash="hash",
        canonical_text_normalized_length=16,
        canonical_text_raw_length=16,
        parents=(parent,),
        metadata={},
    )


def test_profile_accumulates_document_statistics() -> None:
    """Profiling should count documents and parents."""

    accumulator = ParserProfileAccumulator(
        dataset_id="synthetic"
    )

    accumulator.add(make_parsed_document())

    result = accumulator.to_dict()

    assert result["document_count"] == 1
    assert result["parent_count"] == 1
    assert result[
        "single_parent_document_count"
    ] == 1
    assert result["heading_parent_count"] == 1
    assert result[
        "average_parents_per_document"
    ] == 1.0


def test_empty_profile_uses_zero_values() -> None:
    """An empty profile should remain serializable."""

    accumulator = ParserProfileAccumulator(
        dataset_id="synthetic"
    )

    result = accumulator.to_dict()

    assert result["document_count"] == 0
    assert result["parent_count"] == 0
    assert result[
        "median_document_characters"
    ] == 0.0