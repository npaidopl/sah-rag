from __future__ import annotations

import re
from typing import Any

import pytest

from rag_system.chunking import (
    HuggingFaceOffsetTokenizer,
)


class FakeFastTokenizer:
    """Minimal fast-tokenizer-compatible test double."""

    is_fast = True
    model_max_length = 8192

    def num_special_tokens_to_add(
        self,
        pair: bool,
    ) -> int:
        """Return two sequence-level special tokens."""
        assert pair is False
        return 2

    def __call__(
        self,
        text: str,
        **kwargs: Any,
    ) -> dict[str, object]:
        """Return word offsets in Hugging Face format."""
        assert kwargs["add_special_tokens"] is False
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["truncation"] is False

        matches = list(
            re.finditer(r"\S+", text)
        )
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [
                (m.start(), m.end()) for m in matches
            ],
        }


class FakeSlowTokenizer(FakeFastTokenizer):
    """Tokenizer without fast offset support."""

    is_fast = False


def test_adapter_preserves_offsets() -> None:
    """Adapted tokens must resolve to exact source slices."""
    adapter = HuggingFaceOffsetTokenizer(
        FakeFastTokenizer(),
        model_id="BAAI/bge-m3",
        revision="abc123",
    )
    text = "Άρθρο 5 δικαιώματα"
    tokens = adapter.tokenize_with_offsets(text)
    assert [
        token.text for token in tokens
    ] == [
        "Άρθρο",
        "5",
        "δικαιώματα",
    ]
    for token in tokens:
        assert text[token.start:token.end] == token.text


def test_adapter_reports_special_tokens() -> None:
    """Sequence-level special-token count should be exposed."""
    adapter = HuggingFaceOffsetTokenizer(
        FakeFastTokenizer(),
        model_id="test/model",
    )
    assert adapter.special_tokens_count == 2
    assert adapter.model_max_length == 8192


def test_adapter_identity_includes_revision() -> None:
    """Tokenizer identity must include model and revision."""
    adapter = HuggingFaceOffsetTokenizer(
        FakeFastTokenizer(),
        model_id="BAAI/bge-m3",
        revision="revision-001",
    )
    assert adapter.tokenizer_id == (
        "huggingface:BAAI/bge-m3"
        "@revision-001:fast"
    )


def test_slow_tokenizer_is_rejected() -> None:
    """Slow tokenizers cannot guarantee offset mappings."""
    with pytest.raises(ValueError):
        HuggingFaceOffsetTokenizer(
            FakeSlowTokenizer(),
            model_id="test/model",
        )


def test_empty_model_id_is_rejected() -> None:
    """Tokenizer model identity cannot be empty."""
    with pytest.raises(ValueError):
        HuggingFaceOffsetTokenizer(
            FakeFastTokenizer(),
            model_id=" ",
        )


def test_bge_style_budget_reserves_special_tokens() -> None:
    """A 256-token model budget leaves 254 content tokens."""

    adapter = HuggingFaceOffsetTokenizer(
        FakeFastTokenizer(),
        model_id="BAAI/bge-m3",
    )

    from rag_system.chunking import (
        ChildChunkerConfig,
    )

    config = ChildChunkerConfig(
        chunk_size=256,
        chunk_overlap=32,
    )

    assert config.content_chunk_size(
        special_tokens_count=(
            adapter.special_tokens_count
        )
    ) == 254

def test_model_budget_rejects_insufficient_capacity() -> None:
    """The model budget must exceed special-token overhead."""

    from rag_system.chunking import ChildChunkerConfig

    config = ChildChunkerConfig(
        chunk_size=2,
        chunk_overlap=0,
    )

    with pytest.raises(ValueError):
        config.content_chunk_size(
            special_tokens_count=2
        )
