"""Offline tests for native_e2e.context_builder."""

from __future__ import annotations

import pytest

from baselines.base import RetrievedChunk
from baselines.longrag_v2.native_e2e.context_builder import build_context


def _chunk(doc_id: str, text: str, rank: int = 1, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, chunk_id=doc_id, text=text, score=score, rank=rank
    )


def test_empty_returns_empty() -> None:
    ctx, n = build_context([])
    assert ctx == ""
    assert n == 0


def test_single_chunk() -> None:
    ctx, n = build_context([_chunk("d1", "hello")])
    assert ctx == "hello"
    assert n == 1


def test_dedupes_by_text() -> None:
    chunks = [
        _chunk("d1", "same"),
        _chunk("d2", "same"),   # duplicate text (same unit, different doc)
        _chunk("d3", "diff"),
    ]
    ctx, n = build_context(chunks)
    assert n == 2
    assert ctx.count("same") == 1
    assert "diff" in ctx


def test_custom_separator() -> None:
    chunks = [_chunk("d1", "a"), _chunk("d2", "b")]
    ctx, n = build_context(chunks, separator=" | ")
    assert ctx == "a | b"
    assert n == 2


def test_max_chars_respected() -> None:
    # Distinct texts so dedupe does not interfere with the size test.
    chunks = [_chunk(f"d{i}", chr(ord("a") + i) * 100) for i in range(5)]
    ctx, n = build_context(chunks, max_chars=250, separator="")
    # Two chunks of 100 chars fit under 250; the third would exceed.
    assert n == 2
    assert len(ctx) <= 250


def test_first_chunk_truncated_when_oversize() -> None:
    chunks = [_chunk("d1", "x" * 500)]
    ctx, n = build_context(chunks, max_chars=100)
    assert n == 1
    assert len(ctx) == 100


def test_invalid_max_chars_raises() -> None:
    with pytest.raises(ValueError):
        build_context([_chunk("d1", "x")], max_chars=0)


def test_empty_text_chunks_skipped() -> None:
    chunks = [_chunk("d1", ""), _chunk("d2", "hello")]
    ctx, n = build_context(chunks)
    assert ctx == "hello"
    assert n == 1