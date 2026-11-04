"""Tests for sah_rag.generation.pipeline (end-to-end, offline)."""

from __future__ import annotations

import pytest

from sah_rag.generation.adapters import DeterministicDryRunAdapter
from sah_rag.generation.answers import GeneratedAnswer
from sah_rag.generation.pipeline import (
    PipelineConfig,
    answer,
)
from sah_rag.retrieval import RetrievedChunk


# --------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------


def _make_chunk(
    *,
    chunk_id: str = "chunk-1",
    parent_id: str = "parent-1",
    doc_id: str = "doc-1",
    dataset: str = "clerk",
    rank: int = 1,
    score: float = 0.9,
    text: str = "The maximum penalty is ten years imprisonment.",
    method: str = "hybrid_rrf",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        doc_id=doc_id,
        dataset=dataset,
        rank=rank,
        score=score,
        text_normalized=text,
        retrieval_method=method,
        payload={},
    )


def _make_chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        _make_chunk(
            chunk_id=f"chunk-{i}",
            parent_id=f"parent-{i}",
            doc_id=f"doc-{i}",
            rank=i,
            score=1.0 - (i - 1) * 0.1,
            text=f"This is source number {i} with content.",
        )
        for i in range(1, n + 1)
    ]


def _dry_run_runtime(answer_text: str = "The answer is ten years [C1].") -> DeterministicDryRunAdapter:
    return DeterministicDryRunAdapter(answer_text=answer_text)


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------


def test_answer_returns_generated_answer():
    result = answer(
        query_id="q-1",
        query_text="What is the maximum penalty?",
        dataset_id="clerk",
        retrieved_chunks=_make_chunks(3),
        runtime=_dry_run_runtime(),
        config=PipelineConfig(),
    )

    assert isinstance(result, GeneratedAnswer)
    assert result.dataset_id == "clerk"
    assert result.query_id == "q-1"
    assert result.generation_model_id == "Qwen/Qwen2.5-14B-Instruct"
    assert result.generation_runtime == "deterministic-dry-run"
    assert result.citation_validation.valid is True
    assert result.citation_validation.used_citation_ids == ("[C1]",)


def test_answer_empty_chunks_raises():
    with pytest.raises(ValueError, match="retrieved_chunks cannot be empty"):
        answer(
            query_id="q-1",
            query_text="?",
            dataset_id="clerk",
            retrieved_chunks=[],
            runtime=_dry_run_runtime(),
        )


def test_answer_empty_query_id_raises():
    with pytest.raises(ValueError, match="query_id cannot be empty"):
        answer(
            query_id="",
            query_text="?",
            dataset_id="clerk",
            retrieved_chunks=_make_chunks(1),
            runtime=_dry_run_runtime(),
        )


def test_answer_dataset_mismatch_raises():
    chunks = _make_chunks(2)  # dataset="clerk"
    with pytest.raises(ValueError, match="does not match chunk dataset"):
        answer(
            query_id="q-1",
            query_text="?",
            dataset_id="cuad",
            retrieved_chunks=chunks,
            runtime=_dry_run_runtime(),
        )


def test_answer_mixed_datasets_raises():
    chunk_a = _make_chunk(chunk_id="c-1", dataset="clerk")
    chunk_b = _make_chunk(chunk_id="c-2", dataset="cuad")
    with pytest.raises(ValueError, match="same dataset"):
        answer(
            query_id="q-1",
            query_text="?",
            dataset_id="clerk",
            retrieved_chunks=[chunk_a, chunk_b],
            runtime=_dry_run_runtime(),
        )


def test_answer_invalid_citation_raises():
    """Answer with [C99] when prompt only has [C1] should be rejected."""

    bad_runtime = _dry_run_runtime(answer_text="Answer with fake source [C99].")

    with pytest.raises(ValueError, match="only known citations"):
        answer(
            query_id="q-1",
            query_text="?",
            dataset_id="clerk",
            retrieved_chunks=_make_chunks(2),
            runtime=bad_runtime,
        )


def test_pipeline_config_validation():
    with pytest.raises(ValueError, match="positive"):
        PipelineConfig(maximum_context_tokens=0)

    with pytest.raises(ValueError, match="must be >="):
        PipelineConfig(reserved_answer_tokens=-5)

    with pytest.raises(ValueError, match="must be < maximum_prompt_tokens"):
        PipelineConfig(maximum_prompt_tokens=100, reserved_answer_tokens=100)


def test_answer_custom_token_counter():
    """Custom token counter is honored."""

    def char_counter(s: str) -> int:
        return max(1, len(s) // 4)

    result = answer(
        query_id="q-1",
        query_text="What is the penalty?",
        dataset_id="clerk",
        retrieved_chunks=_make_chunks(2),
        runtime=_dry_run_runtime(),
        config=PipelineConfig(),
        token_counter=char_counter,
    )
    assert isinstance(result, GeneratedAnswer)
    # Tokenizer ID propagates from config
    assert result.prompt_id.startswith("prompt-clerk-q-1-")


def test_answer_all_chunks_filtered_by_budget_raises():
    """If no chunk fits in the token budget, raise a clear error
    BEFORE trying to build the prompt (better UX than generic
    'Generation-prompt fields cannot be empty')."""

    with pytest.raises(ValueError, match="No chunks fit within the token budget"):
        answer(
            query_id="q-1",
            query_text="?",
            dataset_id="clerk",
            retrieved_chunks=_make_chunks(1),
            runtime=_dry_run_runtime(answer_text="Answer [C1]."),
            config=PipelineConfig(
                maximum_context_tokens=1,  # too small for any chunk
                maximum_prompt_tokens=100,
                reserved_answer_tokens=20,
            ),
        )