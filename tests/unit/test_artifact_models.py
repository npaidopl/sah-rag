"""Tests for parsed-document and child-chunk artifact models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sah_rag.artifacts.models import (
    BoundaryEvidence,
    BoundaryMethod,
    ChildChunk,
    OffsetSpan,
    ParsedDocument,
    SemanticParent,
)


def make_parent(
    *,
    parent_id: str = "parent-001",
    parent_parent_id: str | None = None,
    hierarchy_level: int = 0,
    start: int = 0,
    end: int = 100,
) -> SemanticParent:
    """Return a valid semantic parent."""

    return SemanticParent(
        parent_id=parent_id,
        doc_id="doc-001",
        dataset="test_dataset",
        parent_parent_id=parent_parent_id,
        hierarchy_level=hierarchy_level,
        sibling_index=0,
        title="Example section",
        text_normalized="A" * (end - start),
        text_raw=None,
        normalized_char_span=OffsetSpan(
            start=start,
            end=end,
        ),
        raw_char_span=None,
        token_span=None,
        token_count=20,
        start_boundary=(
            BoundaryEvidence(
                method=BoundaryMethod.DOCUMENT_START,
                confidence=1.0,
            ),
        ),
        end_boundary=(
            BoundaryEvidence(
                method=BoundaryMethod.PARAGRAPH_BREAK,
                confidence=0.9,
            ),
        ),
        parser_id="deterministic_parser",
        parser_version="0.1.0",
        parser_config_hash="parser-hash",
        metadata={},
    )


def make_chunk() -> ChildChunk:
    """Return a valid child chunk."""

    return ChildChunk(
        chunk_id="chunk-001",
        parent_id="parent-001",
        doc_id="doc-001",
        dataset="test_dataset",
        chunk_index=0,
        parent_chunk_index=0,
        text_normalized="Example chunk.",
        text_raw=None,
        normalized_char_span=OffsetSpan(
            start=10,
            end=24,
        ),
        parent_normalized_char_span=OffsetSpan(
            start=0,
            end=100,
        ),
        raw_char_span=None,
        token_span=OffsetSpan(
            start=0,
            end=3,
        ),
        token_count=3,
        parser_id="deterministic_parser",
        parser_version="0.1.0",
        parser_config_hash="parser-hash",
        chunker_id="token_chunker",
        chunker_version="0.1.0",
        chunker_config_hash="chunker-hash",
        tokenizer_id="test-tokenizer",
        chunk_size=256,
        chunk_overlap=32,
        metadata={},
    )


def test_offset_span_length() -> None:
    """Half-open spans should expose their length."""

    span = OffsetSpan(start=5, end=17)

    assert span.length == 12


def test_offset_span_rejects_reversed_span() -> None:
    """A reversed offset span should fail."""

    with pytest.raises(ValidationError):
        OffsetSpan(start=10, end=2)


def test_semantic_parent_accepts_valid_record() -> None:
    """A valid semantic parent should validate."""

    parent = make_parent()

    assert parent.parent_id == "parent-001"
    assert parent.normalized_char_span.length == 100


def test_semantic_parent_rejects_self_parent() -> None:
    """A parent cannot point to itself."""

    with pytest.raises(ValidationError):
        make_parent(
            parent_id="parent-001",
            parent_parent_id="parent-001",
        )


def test_parsed_document_accepts_hierarchy() -> None:
    """A parsed document should accept resolvable hierarchy links."""

    root_parent = make_parent(parent_id="parent-root")

    child_parent = make_parent(
        parent_id="parent-child",
        parent_parent_id="parent-root",
        hierarchy_level=1,
    )

    document = ParsedDocument(
        doc_id="doc-001",
        dataset="test_dataset",
        parser_id="deterministic_parser",
        parser_version="0.1.0",
        parser_config_hash="parser-hash",
        canonical_text_normalized_length=100,
        canonical_text_raw_length=100,
        parents=(root_parent, child_parent),
        source_fingerprint=None,
        metadata={},
    )

    assert len(document.parents) == 2


def test_parsed_document_rejects_duplicate_parent_ids() -> None:
    """Duplicate parent identifiers should fail."""

    parent = make_parent()

    with pytest.raises(ValidationError):
        ParsedDocument(
            doc_id="doc-001",
            dataset="test_dataset",
            parser_id="deterministic_parser",
            parser_version="0.1.0",
            parser_config_hash="parser-hash",
            canonical_text_normalized_length=100,
            canonical_text_raw_length=100,
            parents=(parent, parent),
        )


def test_parsed_document_rejects_unknown_parent_link() -> None:
    """Hierarchy links must resolve within the parsed document."""

    parent = make_parent(
        parent_parent_id="missing-parent",
        hierarchy_level=1,
    )

    with pytest.raises(ValidationError):
        ParsedDocument(
            doc_id="doc-001",
            dataset="test_dataset",
            parser_id="deterministic_parser",
            parser_version="0.1.0",
            parser_config_hash="parser-hash",
            canonical_text_normalized_length=100,
            canonical_text_raw_length=100,
            parents=(parent,),
        )


def test_child_chunk_accepts_valid_record() -> None:
    """A valid child chunk should validate."""

    chunk = make_chunk()

    assert chunk.chunk_size == 256
    assert chunk.parent_id == "parent-001"


def test_child_chunk_rejects_invalid_overlap() -> None:
    """Overlap must remain smaller than configured chunk size."""

    raw_chunk = make_chunk().model_dump()
    raw_chunk["chunk_overlap"] = 256

    with pytest.raises(ValidationError):
        ChildChunk.model_validate(raw_chunk)


def test_child_chunk_rejects_span_outside_parent() -> None:
    """A child span must remain inside its parent span."""

    raw_chunk = make_chunk().model_dump()
    raw_chunk["normalized_char_span"] = {
        "start": 90,
        "end": 120,
    }

    with pytest.raises(ValidationError):
        ChildChunk.model_validate(raw_chunk)


def test_parsed_document_rejects_parent_span_past_document() -> None:
    """Parent spans cannot exceed canonical document limits."""

    parent = make_parent(
        start=0,
        end=120,
    )

    with pytest.raises(ValidationError):
        ParsedDocument(
            doc_id="doc-001",
            dataset="test_dataset",
            parser_id="deterministic_parser",
            parser_version="0.1.0",
            parser_config_hash="parser-hash",
            canonical_text_normalized_length=100,
            canonical_text_raw_length=100,
            parents=(parent,),
        )


def test_child_chunk_rejects_token_count_mismatch() -> None:
    """Token count must agree with the recorded token span."""

    raw_chunk = make_chunk().model_dump()
    raw_chunk["token_count"] = 4

    with pytest.raises(ValidationError):
        ChildChunk.model_validate(raw_chunk)