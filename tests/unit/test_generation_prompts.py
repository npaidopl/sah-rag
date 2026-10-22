"""Tests for cited generation prompt artifacts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sah_rag.generation import (
    ContextItem,
    ContextPack,
    build_generation_prompt,
    render_context,
)


def make_context_pack() -> ContextPack:
    """Create one valid context pack."""

    items = (
        ContextItem(
            citation_id="[C1]",
            chunk_id="chunk-a",
            parent_id="parent-a",
            doc_id="doc-a",
            dataset="synthetic",
            retrieval_rank=1,
            retrieval_score=0.9,
            retrieval_method="hybrid_rrf",
            text="Alpha source text.",
            token_count=3,
        ),
        ContextItem(
            citation_id="[C2]",
            chunk_id="chunk-b",
            parent_id="parent-b",
            doc_id="doc-b",
            dataset="synthetic",
            retrieval_rank=2,
            retrieval_score=0.8,
            retrieval_method="hybrid_rrf",
            text="Beta source text.",
            token_count=3,
        ),
    )

    return ContextPack(
        context_pack_id="context-synthetic-query-001",
        query_id="query-001",
        query_text="What is supported?",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=10,
        used_context_tokens=6,
        remaining_context_tokens=4,
        selected_item_count=2,
        skipped_item_count=0,
        selected_items=items,
        skipped_chunk_ids=(),
    )


def count_words(text: str) -> int:
    """Count whitespace-delimited tokens."""

    return len(text.split())


def test_rendered_context_preserves_citations() -> None:
    """Every context item must retain its citation."""

    rendered = render_context(make_context_pack())

    assert "[C1]" in rendered
    assert "[C2]" in rendered
    assert "doc-a" in rendered
    assert "chunk-b" in rendered


def test_generation_prompt_is_deterministic() -> None:
    """Equivalent context must produce equal prompts."""

    context_pack = make_context_pack()

    first = build_generation_prompt(
        context_pack=context_pack,
        token_counter=count_words,
        maximum_prompt_tokens=256,
        reserved_answer_tokens=32,
    )

    second = build_generation_prompt(
        context_pack=context_pack,
        token_counter=count_words,
        maximum_prompt_tokens=256,
        reserved_answer_tokens=32,
    )

    assert first == second
    assert first.citation_ids == ("[C1]", "[C2]")


def test_prompt_budget_reserves_answer_tokens() -> None:
    """Rendered prompts must respect answer reservations."""

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        build_generation_prompt(
            context_pack=make_context_pack(),
            token_counter=count_words,
            maximum_prompt_tokens=20,
            reserved_answer_tokens=10,
        )


def test_prompt_rejects_invalid_citation_declaration() -> None:
    """Declared citations must match rendered citations."""

    prompt = build_generation_prompt(
        context_pack=make_context_pack(),
        token_counter=count_words,
        maximum_prompt_tokens=256,
        reserved_answer_tokens=32,
    )

    with pytest.raises(
        ValidationError,
        match="Rendered citations",
    ):
        prompt.__class__.model_validate(
            {
                **prompt.model_dump(mode="json"),
                "citation_ids": ["[C1]", "[C3]"],
            }
        )
