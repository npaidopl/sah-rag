"""Tests for offset-preserving tokenization."""

from __future__ import annotations

from sah_rag.chunking import RegexOffsetTokenizer


def test_regex_tokenizer_preserves_offsets() -> None:
    """Every token must map exactly to the source string."""

    text = "Article 5: This is a test."

    tokenizer = RegexOffsetTokenizer()
    tokens = tokenizer.tokenize_with_offsets(text)

    assert tokens

    for token in tokens:
        assert text[token.start:token.end] == token.text


def test_regex_tokenizer_is_deterministic() -> None:
    """Repeated tokenization should be identical."""

    text = "Dense + sparse retrieval."

    tokenizer = RegexOffsetTokenizer()

    assert (
        tokenizer.tokenize_with_offsets(text)
        == tokenizer.tokenize_with_offsets(text)
    )


def test_regex_tokenizer_handles_unicode() -> None:
    """Greek text should retain exact Unicode slices."""

    text = "Άρθρο 5: Δικαιώματα και υποχρεώσεις."

    tokenizer = RegexOffsetTokenizer()
    tokens = tokenizer.tokenize_with_offsets(text)

    assert [
        token.text
        for token in tokens
    ][:3] == [
        "Άρθρο",
        "5",
        ":",
    ]

    for token in tokens:
        assert text[token.start:token.end] == token.text


def test_regex_tokenizer_handles_empty_text() -> None:
    """Empty text should produce no tokens."""

    tokenizer = RegexOffsetTokenizer()

    assert tokenizer.tokenize_with_offsets("") == ()
