"""Tests for the first deterministic structural parser."""

from __future__ import annotations

import json
from pathlib import Path

from sah_rag.canonical.models import CanonicalDocument
from sah_rag.parsing import (
    DeterministicStructuralParser,
    iter_paragraph_spans,
)

FIXTURE_PATH = Path(
    "tests/fixtures/structured_document.json"
)


def load_fixture() -> CanonicalDocument:
    """Load the synthetic structured document."""

    raw_record = json.loads(
        FIXTURE_PATH.read_text(encoding="utf-8-sig")
    )

    return CanonicalDocument.model_validate(raw_record)


def test_paragraph_segmentation_preserves_offsets() -> None:
    """Every block must map exactly to its source slice."""

    document = load_fixture()
    blocks = iter_paragraph_spans(
        document.text_normalized
    )

    assert len(blocks) == 8

    for block in blocks:
        assert (
            document.text_normalized[
                block.start:block.end
            ]
            == block.text
        )


def test_parser_detects_three_structural_parents() -> None:
    """Three explicit headings should create three parents."""

    document = load_fixture()
    parser = DeterministicStructuralParser()

    parsed = parser.parse(document)

    assert len(parsed.parents) == 3

    assert [
        parent.title
        for parent in parsed.parents
    ] == [
        "CHAPTER 1",
        "SECTION 1.1 Definitions",
        "CHAPTER 2",
    ]


def test_parent_spans_preserve_source_text() -> None:
    """Parent text must equal the canonical source slice."""

    document = load_fixture()
    parser = DeterministicStructuralParser()

    parsed = parser.parse(document)

    for parent in parsed.parents:
        span = parent.normalized_char_span

        assert (
            document.text_normalized[
                span.start:span.end
            ]
            == parent.text_normalized
        )


def test_parent_ids_are_deterministic() -> None:
    """Repeated parsing must produce identical parent IDs."""

    document = load_fixture()
    parser = DeterministicStructuralParser()

    first = parser.parse(document)
    second = parser.parse(document)

    assert [
        parent.parent_id
        for parent in first.parents
    ] == [
        parent.parent_id
        for parent in second.parents
    ]


def test_unstructured_document_remains_one_parent() -> None:
    """Unstructured text should remain one semantic parent."""

    text = "\n\n".join(
        [
            "First paragraph.",
            "Second paragraph.",
            "Third paragraph.",
            "Fourth paragraph.",
            "Fifth paragraph.",
        ]
    )

    document = CanonicalDocument(
        doc_id="unstructured-001",
        dataset="synthetic",
        text_raw=text,
        text_normalized=text,
        title=None,
        source_path=None,
        source_row=None,
        metadata={},
    )

    parser = DeterministicStructuralParser()
    parsed = parser.parse(document)

    assert len(parsed.parents) == 1
    assert (
        parsed.parents[0].text_normalized
        == text
    )
    assert parsed.parents[0].title is None


def test_small_orphan_heading_merges_forward() -> None:
    """A heading-only orphan should merge into the next section."""

    text = "\n\n".join(
        [
            "ANNEX V",
            "SECTION 1 Main Terms",
            "This section contains the actual body text.",
        ]
    )

    document = CanonicalDocument(
        doc_id="orphan-heading-001",
        dataset="synthetic",
        text_raw=text,
        text_normalized=text,
        title=None,
        source_path=None,
        source_row=None,
        metadata={},
    )

    parser = DeterministicStructuralParser()
    parsed = parser.parse(document)

    assert len(parsed.parents) == 1
    assert "ANNEX V" in parsed.parents[0].text_normalized
    assert "actual body text" in (
        parsed.parents[0].text_normalized
    )


def test_empty_document_produces_no_parents() -> None:
    """An empty canonical document should remain valid."""

    document = CanonicalDocument(
        doc_id="empty-001",
        dataset="synthetic",
        text_raw="",
        text_normalized="",
        title=None,
        source_path=None,
        source_row=None,
        metadata={},
    )

    parser = DeterministicStructuralParser()
    parsed = parser.parse(document)

    assert parsed.parents == ()
    assert parsed.metadata["block_count"] == 0
    assert parsed.metadata["parent_count"] == 0