"""Tests for provider-neutral generation runtimes."""

from __future__ import annotations

from rag_system.generation import (
    ContextItem,
    ContextPack,
    DeterministicDryRunAdapter,
    GenerationRequest,
    build_generated_answer,
    build_generation_prompt,
)


def count_words(text: str) -> int:
    """Count test tokens."""

    return len(text.split())


def make_prompt():
    """Create one runtime test prompt."""

    context_pack = ContextPack(
        context_pack_id="context-runtime-test",
        query_id="query-001",
        query_text="What is supported?",
        dataset_id="synthetic",
        retrieval_method="hybrid_rrf",
        tokenizer_id="test-tokenizer",
        maximum_context_tokens=20,
        used_context_tokens=3,
        remaining_context_tokens=17,
        selected_item_count=1,
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
        ),
        skipped_chunk_ids=(),
    )

    return build_generation_prompt(
        context_pack=context_pack,
        token_counter=count_words,
        maximum_prompt_tokens=256,
        reserved_answer_tokens=32,
    )


def test_dry_run_adapter_returns_normalized_response() -> None:
    """Dry-run adapters should satisfy the runtime contract."""

    prompt = make_prompt()

    request = GenerationRequest(
        prompt=prompt,
        model_id="synthetic-model",
        maximum_output_tokens=32,
        temperature=0.0,
        seed=123,
    )

    adapter = DeterministicDryRunAdapter(
        answer_text=(
            "The source supports the answer [C1]."
        )
    )

    response = adapter.generate(request)

    assert response.runtime_id == (
        "deterministic-dry-run"
    )

    assert response.model_id == (
        "synthetic-model"
    )

    assert response.input_token_count == (
        prompt.rendered_prompt_tokens
    )


def test_runtime_response_builds_valid_answer() -> None:
    """Normalized responses should produce typed answers."""

    prompt = make_prompt()

    request = GenerationRequest(
        prompt=prompt,
        model_id="synthetic-model",
        maximum_output_tokens=32,
        temperature=0.0,
        seed=123,
    )

    adapter = DeterministicDryRunAdapter(
        answer_text=(
            "The source supports the answer [C1]."
        )
    )

    response = adapter.generate(request)

    answer = build_generated_answer(
        request=request,
        response=response,
        metadata={
            "artifact_validation_only": True,
        },
    )

    assert answer.citation_validation.valid is True
    assert answer.generation_runtime == (
        "deterministic-dry-run"
    )

    assert answer.metadata[
        "runtime_request_id"
    ] is None

    assert answer.metadata[
        "artifact_validation_only"
    ] is True
