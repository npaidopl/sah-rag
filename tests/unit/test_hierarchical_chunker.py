"""Tests for parent-aware child chunking."""

from __future__ import annotations

from sah_rag.artifacts import (
    BoundaryEvidence,
    BoundaryMethod,
    OffsetSpan,
    ParsedDocument,
    SemanticParent,
)
from sah_rag.chunking import (
    ChildChunkerConfig,
    HierarchicalChildChunker,
    RegexOffsetTokenizer,
)


def make_parent(
    *,
    parent_id: str,
    sibling_index: int,
    text: str,
    start: int,
) -> SemanticParent:
    """Create one valid semantic parent."""

    end = start + len(text)

    return SemanticParent(
        parent_id=parent_id,
        doc_id="doc-001",
        dataset="synthetic",
        parent_parent_id=None,
        hierarchy_level=0,
        sibling_index=sibling_index,
        title=None,
        text_normalized=text,
        text_raw=text,
        normalized_char_span=OffsetSpan(
            start=start,
            end=end,
        ),
        raw_char_span=OffsetSpan(
            start=start,
            end=end,
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
        parser_version="0.3.0",
        parser_config_hash="parser-hash",
        metadata={},
    )


def make_document() -> ParsedDocument:
    """Create a document with two semantic parents."""

    first_text = "one two three four five six"
    second_text = "seven eight nine ten"

    first = make_parent(
        parent_id="parent-001",
        sibling_index=0,
        text=first_text,
        start=0,
    )

    second_start = len(first_text) + 2

    second = make_parent(
        parent_id="parent-002",
        sibling_index=1,
        text=second_text,
        start=second_start,
    )

    return ParsedDocument(
        doc_id="doc-001",
        dataset="synthetic",
        parser_id="parser",
        parser_version="0.3.0",
        parser_config_hash="parser-hash",
        canonical_text_normalized_length=(
            second_start + len(second_text)
        ),
        canonical_text_raw_length=(
            second_start + len(second_text)
        ),
        parents=(first, second),
        metadata={},
    )


def test_chunker_preserves_parent_boundaries() -> None:
    """No chunk may contain text from two parents."""

    document = make_document()

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 3

    assert [
        chunk.parent_id
        for chunk in chunks
    ] == [
        "parent-001",
        "parent-001",
        "parent-002",
    ]


def test_chunker_creates_expected_overlap() -> None:
    """Sliding windows should overlap by configured tokens."""

    document = make_document()

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    chunks = chunker.chunk_document(document)

    first_parent_chunks = [
        chunk
        for chunk in chunks
        if chunk.parent_id == "parent-001"
    ]

    assert [
        chunk.token_span.model_dump()
        for chunk in first_parent_chunks
    ] == [
        {"start": 0, "end": 4},
        {"start": 3, "end": 6},
    ]


def test_chunk_text_matches_parent_slice() -> None:
    """Chunk offsets must resolve to the parent source text."""

    document = make_document()

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=3,
            chunk_overlap=1,
        ),
    )

    chunks = chunker.chunk_document(document)

    parents = {
        parent.parent_id: parent
        for parent in document.parents
    }

    for chunk in chunks:
        parent = parents[chunk.parent_id]

        relative_start = (
            chunk.normalized_char_span.start
            - parent.normalized_char_span.start
        )
        relative_end = (
            chunk.normalized_char_span.end
            - parent.normalized_char_span.start
        )

        assert (
            parent.text_normalized[
                relative_start:relative_end
            ]
            == chunk.text_normalized
        )


def test_chunk_ids_are_deterministic() -> None:
    """Repeated chunking must produce stable IDs."""

    document = make_document()

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert [
        chunk.chunk_id
        for chunk in first
    ] == [
        chunk.chunk_id
        for chunk in second
    ]


def test_chunk_indexes_are_contiguous() -> None:
    """Global chunk indexes should cover one contiguous range."""

    document = make_document()

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=2,
            chunk_overlap=0,
        ),
    )

    chunks = chunker.chunk_document(document)

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == list(range(len(chunks)))

def test_chunker_avoids_tiny_final_tail() -> None:
    """Final windows should be anchored when the tail is tiny."""

    text = " ".join(
        f"token{index}"
        for index in range(15)
    )

    parent = make_parent(
        parent_id="parent-tail",
        sibling_index=0,
        text=text,
        start=0,
    )

    document = ParsedDocument(
        doc_id="doc-001",
        dataset="synthetic",
        parser_id="parser",
        parser_version="0.3.0",
        parser_config_hash="parser-hash",
        canonical_text_normalized_length=len(text),
        canonical_text_raw_length=len(text),
        parents=(parent,),
        metadata={},
    )

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=8,
            chunk_overlap=2,
        ),
    )

    chunks = chunker.chunk_document(document)

    assert [
        chunk.token_span.model_dump()
        for chunk in chunks
    ] == [
        {"start": 0, "end": 8},
        {"start": 6, "end": 14},
        {"start": 7, "end": 15},
    ]

    assert all(
        chunk.token_count >= 4
        for chunk in chunks
    )


def test_chunk_metadata_records_model_budget() -> None:
    """Chunks should record capacity and special-token policy."""

    document = make_document()

    chunker = HierarchicalChildChunker(
        tokenizer=RegexOffsetTokenizer(),
        config=ChildChunkerConfig(
            chunk_size=4,
            chunk_overlap=1,
        ),
    )

    chunks = chunker.chunk_document(document)
    first = chunks[0]

    assert first.metadata[
        "model_input_token_budget"
    ] == 4

    assert first.metadata[
        "content_token_capacity"
    ] == 4

    assert first.metadata[
        "special_tokens_count"
    ] == 0

    assert first.metadata[
        "estimated_model_input_tokens"
    ] == first.token_count
