"""Tests for deterministic heading provenance."""
from __future__ import annotations

from rag_system.parsing import (
    DeterministicParserConfig,
    HeadingReason,
    TextBlock,
    detect_heading_reason,
)

CONFIG = DeterministicParserConfig()


def make_block(text: str) -> TextBlock:
    """Create one exact text block."""
    return TextBlock(
        block_index=0,
        text=text,
        start=0,
        end=len(text),
    )


def test_numbered_heading_reason() -> None:
    """Numbered heading rules should identify their provenance."""
    reason = detect_heading_reason(
        make_block("1. Introduction"),
        CONFIG,
    )
    assert reason is HeadingReason.NUMBERED


def test_named_heading_reason() -> None:
    """Named headings should be identified."""
    reason = detect_heading_reason(
        make_block("CHAPTER One"),
        CONFIG,
    )
    assert reason is HeadingReason.NAMED


def test_uppercase_heading_reason() -> None:
    """Short uppercase blocks should use the uppercase rule."""
    reason = detect_heading_reason(
        make_block("GENERAL CONDITIONS"),
        CONFIG,
    )
    assert reason is HeadingReason.UPPERCASE


def test_regular_sentence_is_not_heading() -> None:
    """Normal sentence text should not be a heading."""
    reason = detect_heading_reason(
        make_block(
            "This sentence contains ordinary body text."
        ),
        CONFIG,
    )
    assert reason is None

def test_bare_year_is_not_numbered_heading() -> None:
    """A year followed by table data is not a heading."""

    reason = detect_heading_reason(
        make_block("2024 2023"),
        CONFIG,
    )

    assert reason is None


def test_address_is_not_numbered_heading() -> None:
    """A street address is not a numbered heading."""

    reason = detect_heading_reason(
        make_block(
            "1680 38th Street, Suite 130 80301"
        ),
        CONFIG,
    )

    assert reason is None


def test_sentence_starting_with_one_is_not_heading() -> None:
    """OCR text beginning with digit one is body text."""

    reason = detect_heading_reason(
        make_block(
            "1 have no knickers, all my clothes are small,"
        ),
        CONFIG,
    )

    assert reason is None


def test_equation_is_not_uppercase_heading() -> None:
    """Equations must not become uppercase headings."""

    reason = detect_heading_reason(
        make_block("P*A = L*U"),
        CONFIG,
    )

    assert reason is None


def test_named_word_inside_sentence_is_not_heading() -> None:
    """A sentence mentioning a chapter is not a heading."""

    reason = detect_heading_reason(
        make_block(
            "Chapter two was found, and she read it."
        ),
        CONFIG,
    )

    assert reason is None


def test_article_heading_remains_supported() -> None:
    """A compact legal article heading remains valid."""

    reason = detect_heading_reason(
        make_block(
            "Article 3.Governing Law"
        ),
        CONFIG,
    )

    assert reason is HeadingReason.NAMED