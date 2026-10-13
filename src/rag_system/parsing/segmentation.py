"""Deterministic paragraph segmentation with exact offsets."""

from __future__ import annotations

import re

from rag_system.parsing.blocks import TextBlock

_PARAGRAPH_SEPARATOR = re.compile(
    r"(?:\r?\n[ \t]*){2,}"
)


def iter_paragraph_spans(
    text: str,
) -> tuple[TextBlock, ...]:
    """Split text into non-empty blocks while preserving offsets."""

    if not text:
        return ()

    blocks: list[TextBlock] = []
    cursor = 0

    for separator in _PARAGRAPH_SEPARATOR.finditer(text):
        raw_start = cursor
        raw_end = separator.start()

        _append_trimmed_block(
            blocks=blocks,
            source=text,
            raw_start=raw_start,
            raw_end=raw_end,
        )

        cursor = separator.end()

    _append_trimmed_block(
        blocks=blocks,
        source=text,
        raw_start=cursor,
        raw_end=len(text),
    )

    return tuple(blocks)


def _append_trimmed_block(
    *,
    blocks: list[TextBlock],
    source: str,
    raw_start: int,
    raw_end: int,
) -> None:
    """Append one trimmed non-empty source slice."""

    while (
        raw_start < raw_end
        and source[raw_start].isspace()
    ):
        raw_start += 1

    while (
        raw_end > raw_start
        and source[raw_end - 1].isspace()
    ):
        raw_end -= 1

    if raw_start >= raw_end:
        return

    block_text = source[raw_start:raw_end]

    blocks.append(
        TextBlock(
            block_index=len(blocks),
            text=block_text,
            start=raw_start,
            end=raw_end,
        )
    )