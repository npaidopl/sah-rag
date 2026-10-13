"""Structural text blocks used during deterministic parsing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One non-empty paragraph-like text block."""

    block_index: int
    text: str
    start: int
    end: int
    is_heading: bool = False

    def __post_init__(self) -> None:
        """Validate block offsets and text."""

        if self.block_index < 0:
            raise ValueError(
                "block_index cannot be negative."
            )

        if self.start < 0 or self.end < self.start:
            raise ValueError(
                "Text-block offsets are invalid."
            )

        if not self.text:
            raise ValueError(
                "Text blocks cannot contain empty text."
            )

        if len(self.text) != self.end - self.start:
            raise ValueError(
                "Text length must equal the block-span length."
            )