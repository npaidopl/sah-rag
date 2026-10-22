"""Tests for generated answers and citation validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sah_rag.generation import (
    ContextItem,
    ContextPack,
    GeneratedAnswer,
    build_generation_prompt,
    extract_answer_citations,
    validate_answer_citations,
)


def count_words(text: str) -> int:
    """Count simple test tokens."""

    return len(text.split())


def make_prompt():
    """Create one cited generation prompt."""
    context = ContextPack(
        context_pack_id="context-test",
        query_id="query-001",
        query_text="What is supported?",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=30,
        used_context_tokens=6,
        remaining_context_tokens=24,
        selected_item_count=2,
        skipped_item_count=0,
        selected_items=(
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
        ),
        skipped_chunk_ids=(),
    )

    return build_generation_prompt(
        context_pack=context,
        token_counter=count_words,
        maximum_prompt_tokens=256,
        reserved_answer_tokens=32,
    )


def test_extract_answer_citations_is_ordered() -> None:
    """Citations should retain first-occurrence order."""

    citations = extract_answer_citations(
        "Claim [C2]. Another [C1]. Repeat [C2]."
    )

    assert citations == (
        "[C2]",
        "[C1]",
    )


def test_known_answer_citations_are_valid() -> None:
    """Known prompt citations should validate."""

    report = validate_answer_citations(
        answer_text=(
            "The first claim is supported [C1]. "
            "The second claim is supported [C2]."
        ),
        prompt=make_prompt(),
    )

    assert report.valid is True
    assert report.used_citation_ids == (
        "[C1]",
        "[C2]",
    )
    assert report.unknown_citation_ids == ()


def test_unknown_answer_citation_is_rejected() -> None:
    """Unknown citations must invalidate the report."""

    report = validate_answer_citations(
        answer_text="Unsupported citation [C9].",
        prompt=make_prompt(),
    )

    assert report.valid is False
    assert report.unknown_citation_ids == (
        "[C9]",
    )


def test_uncited_substantive_answer_is_rejected() -> None:
    """Substantive answers require citations."""

    prompt = make_prompt()

    report = validate_answer_citations(
        answer_text="This answer has no citation.",
        prompt=prompt,
    )

    with pytest.raises(
        ValidationError,
        match="must contain only known citations",
    ):
        GeneratedAnswer(
            answer_id="answer-001",
            prompt_id=prompt.prompt_id,
            context_pack_id=(
                prompt.context_pack_id
            ),
            query_id=prompt.query_id,
            dataset_id=prompt.dataset_id,
            answer_text=(
                "This answer has no citation."
            ),
            generation_model_id="test-model",
            generation_runtime="test-runtime",
            input_token_count=10,
            output_token_count=6,
            refused=False,
            insufficient_context=False,
            citation_validation=report,
        )


def test_refusal_may_omit_citations() -> None:
    """A refusal may contain no source citation."""
    prompt = make_prompt()

    report = validate_answer_citations(
        answer_text=(
            "The available sources are insufficient."
        ),
        prompt=prompt,
    )

    answer = GeneratedAnswer(
        answer_id="answer-refusal",
        prompt_id=prompt.prompt_id,
        context_pack_id=(
            prompt.context_pack_id
        ),
        query_id=prompt.query_id,
        dataset_id=prompt.dataset_id,
        answer_text=(
            "The available sources are insufficient."
        ),
        generation_model_id="test-model",
        generation_runtime="test-runtime",
        input_token_count=10,
        output_token_count=6,
        refused=True,
        insufficient_context=True,
        citation_validation=report,
    )

    assert answer.refused is True
    assert answer.citation_validation.valid is False
