"""Tokenizer abstractions with exact character offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """One token with a half-open character span."""

    token_index: int
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate token identity and offsets."""

        if self.token_index < 0:
            raise ValueError(
                "token_index cannot be negative."
            )

        if self.start < 0 or self.end < self.start:
            raise ValueError(
                "Token offsets are invalid."
            )

        if not self.text:
            raise ValueError(
                "Token text cannot be empty."
            )

        if len(self.text) != self.end - self.start:
            raise ValueError(
                "Token text length must match its span."
            )


@runtime_checkable
class OffsetTokenizer(Protocol):
    """Tokenizer interface required by the child chunker."""

    @property
    def tokenizer_id(self) -> str:
        """Return a stable tokenizer identifier."""




    @property
    def special_tokens_count(self) -> int:
        """Return sequence-level special tokens added by the model."""

    def tokenize_with_offsets(
        self,
        text: str,
    ) -> tuple[TokenSpan, ...]:
        """Tokenize text while preserving character offsets."""


class RegexOffsetTokenizer:
    """Dependency-free tokenizer for tests and development."""

    @property
    def tokenizer_id(self) -> str:
        """Return the tokenizer identity."""

        return "regex-unicode-v1"

    def tokenize_with_offsets(
        self,
        text: str,
    ) -> tuple[TokenSpan, ...]:
        """Split words and punctuation with exact offsets."""

        return tuple(
            TokenSpan(
                token_index=index,
                text=match.group(0),
                start=match.start(),
                end=match.end(),
            )
            for index, match in enumerate(
                _TOKEN_PATTERN.finditer(text)
            )
        )
    @property
    def special_tokens_count(self) -> int:
        """The regex tokenizer adds no model-level tokens."""

        return 0
