"""Tests for deterministic generation context assembly."""

from __future__ import annotations

import pytest

from sah_rag.generation import (
    build_context_pack,
)
from sah_rag.retrieval import RetrievedChunk


def make_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    rank: int,
    score: float,
    text: str,
) -> RetrievedChunk:
    """Create one retrieved chunk."""

    return RetrievedChunk(
        chunk_id=chunk_id,
        parent_id=f"parent-{doc_id}",
        doc_id=doc_id,
        dataset="synthetic",
        rank=rank,
        score=score,
        text_normalized=text,
        retrieval_method="hybrid_rrf",
        payload={},
    )


def token_counter(text: str) -> int:
    """Count whitespace-delimited test tokens."""

    return len(text.split())


def test_context_pack_respects_token_budget() -> None:
    """Selected context must not exceed its token budget."""

    chunks = (
        make_chunk(
            chunk_id="chunk-a",
            doc_id="doc-a",
            rank=1,
            score=0.9,
            text="one two three",
        ),
        make_chunk(
            chunk_id="chunk-b",
            doc_id="doc-b",
            rank=2,
            score=0.8,
            text="four five six",
        ),
        make_chunk(
            chunk_id="chunk-c",
            doc_id="doc-c",
            rank=3,
            score=0.7,
            text="seven eight",
        ),
    )

    pack = build_context_pack(
        query_id="query-001",
        query_text="sample query",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=5,
        retrieved_chunks=chunks,
        token_counter=token_counter,
    )

    assert pack.used_context_tokens == 5
    assert pack.remaining_context_tokens == 0

    assert tuple(item.chunk_id for item in pack.selected_items) == (
        "chunk-a",
        "chunk-c",
    )

    assert pack.skipped_chunk_ids == ("chunk-b",)


def test_context_citations_are_contiguous() -> None:
    """Selected sources must use stable citation labels."""

    chunks = (
        make_chunk(
            chunk_id="chunk-a",
            doc_id="doc-a",
            rank=1,
            score=0.9,
            text="alpha",
        ),
        make_chunk(
            chunk_id="chunk-b",
            doc_id="doc-b",
            rank=2,
            score=0.8,
            text="beta",
        ),
    )

    pack = build_context_pack(
        query_id="query-001",
        query_text="sample query",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=10,
        retrieved_chunks=chunks,
        token_counter=token_counter,
    )

    assert tuple(item.citation_id for item in pack.selected_items) == (
        "[C1]",
        "[C2]",
    )


def test_context_pack_is_deterministic() -> None:
    """Equivalent inputs must produce equal context packs."""

    chunks = (
        make_chunk(
            chunk_id="chunk-a",
            doc_id="doc-a",
            rank=1,
            score=0.9,
            text="alpha beta",
        ),
    )

    first = build_context_pack(
        query_id="query-001",
        query_text="sample query",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=10,
        retrieved_chunks=chunks,
        token_counter=token_counter,
    )

    second = build_context_pack(
        query_id="query-001",
        query_text="sample query",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=10,
        retrieved_chunks=chunks,
        token_counter=token_counter,
    )

    assert first == second


def test_per_document_limit_is_enforced() -> None:
    """Context packs may limit evidence per document."""

    chunks = (
        make_chunk(
            chunk_id="chunk-a1",
            doc_id="doc-a",
            rank=1,
            score=0.9,
            text="first",
        ),
        make_chunk(
            chunk_id="chunk-a2",
            doc_id="doc-a",
            rank=2,
            score=0.8,
            text="second",
        ),
        make_chunk(
            chunk_id="chunk-b1",
            doc_id="doc-b",
            rank=3,
            score=0.7,
            text="third",
        ),
    )

    pack = build_context_pack(
        query_id="query-001",
        query_text="sample query",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=10,
        retrieved_chunks=chunks,
        token_counter=token_counter,
        maximum_chunks_per_document=1,
    )

    assert tuple(item.chunk_id for item in pack.selected_items) == (
        "chunk-a1",
        "chunk-b1",
    )

    assert pack.skipped_chunk_ids == ("chunk-a2",)


def test_invalid_token_count_is_rejected() -> None:
    """Token counters must return positive counts."""

    chunk = make_chunk(
        chunk_id="chunk-a",
        doc_id="doc-a",
        rank=1,
        score=0.9,
        text="alpha",
    )

    with pytest.raises(
        ValueError,
        match="non-positive",
    ):
        build_context_pack(
            query_id="query-001",
            query_text="sample query",
            dataset_id="synthetic",
            retrieval_method="hybrid_rrf",
            tokenizer_id="test-tokenizer",
            maximum_context_tokens=10,
            retrieved_chunks=(chunk,),
            token_counter=lambda _: 0,
        )
