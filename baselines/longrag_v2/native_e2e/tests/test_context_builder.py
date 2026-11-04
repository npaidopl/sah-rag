"""Tests for native_e2e.context_builder (context + titles)."""

from __future__ import annotations

import pytest

from baselines.base import RetrievedChunk
from baselines.longrag_v2.native_e2e.context_builder import (
    BuiltContext,
    build_context,
)


def _chunk(doc_id: str, text: str, rank: int = 1, score: float = 1.0,
           metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, chunk_id=doc_id, text=text, score=score, rank=rank,
        metadata=metadata or {},
    )


def test_empty_returns_empty() -> None:
    b = build_context([])
    assert b.context == ""
    assert b.titles == []
    assert b.num_unique_units == 0


def test_single_chunk_title_from_doc_id() -> None:
    b = build_context([_chunk("d1", "hello")])
    assert b.context == "hello"
    assert b.titles == ["d1"]
    assert b.num_unique_units == 1


def test_title_from_metadata_when_present() -> None:
    b = build_context([
        _chunk("d1", "hello", metadata={"title": "Real Title"}),
    ])
    assert b.titles == ["Real Title"]


def test_title_falls_back_when_metadata_title_empty() -> None:
    b = build_context([
        _chunk("d1", "hello", metadata={"title": "   "}),
    ])
    assert b.titles == ["d1"]


def test_dedupes_by_text_keeps_first_title() -> None:
    chunks = [
        _chunk("d1", "same", metadata={"title": "First"}),
        _chunk("d2", "same", metadata={"title": "Second"}),
        _chunk("d3", "diff", metadata={"title": "Third"}),
    ]
    b = build_context(chunks)
    assert b.num_unique_units == 2
    assert b.context.count("same") == 1
    assert "First" in b.titles
    assert "Second" not in b.titles
    assert "Third" in b.titles


def test_custom_separator() -> None:
    b = build_context([_chunk("d1", "a"), _chunk("d2", "b")], separator=" | ")
    assert b.context == "a | b"
    assert b.num_unique_units == 2


def test_max_chars_respected_titles_truncated() -> None:
    chunks = [
        _chunk(f"d{i}", chr(ord("a") + i) * 100)
        for i in range(5)
    ]
    b = build_context(chunks, max_chars=250, separator="")
    assert b.num_unique_units == 2
    assert len(b.context) <= 250
    assert len(b.titles) == 2


def test_first_chunk_truncated_when_oversize() -> None:
    b = build_context([_chunk("d1", "x" * 500)], max_chars=100)
    assert b.num_unique_units == 1
    assert len(b.context) == 100
    assert b.titles == ["d1"]


def test_invalid_max_chars_raises() -> None:
    with pytest.raises(ValueError):
        build_context([_chunk("d1", "x")], max_chars=0)


def test_empty_text_chunks_skipped() -> None:
    b = build_context([_chunk("d1", ""), _chunk("d2", "hello")])
    assert b.context == "hello"
    assert b.titles == ["d2"]
    assert b.num_unique_units == 1


def test_no_duplicate_titles() -> None:
    # Even if two chunks share doc_id (unlikely but possible), titles unique
    chunks = [
        _chunk("d1", "aaa"),
        _chunk("d1", "bbb"),
    ]
    b = build_context(chunks)
    assert b.titles == ["d1"]