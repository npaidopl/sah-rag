"""End-to-end RAG pipeline: retrieved chunks -> generated answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sah_rag.generation.answers import (
    GeneratedAnswer,
    build_generated_answer,
)
from sah_rag.generation.context import (
    TokenCounter,
    build_context_pack,
)
from sah_rag.generation.prompts import (
    build_generation_prompt,
)
from sah_rag.generation.qwen_adapter import QwenLLMAdapter
from sah_rag.generation.runtime import (
    GenerationRequest,
    GenerationRuntimeAdapter,
)
from sah_rag.retrieval import RetrievedChunk


def _default_word_counter(text: str) -> int:
    """Cheap approximate token counter (1 token ~ 1 word).

    This UNDER-estimates the true token count for typical text
    (Qwen tokenizer tends to be 1.2-1.4x word count). The pipeline
    leaves ~400 token headroom by default, which absorbs the error.
    For exact counts, pass token_counter=qwen_tokenizer.encode_call.
    """

    return max(1, len(text.split()))


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen configuration for the end-to-end RAG pipeline."""

    maximum_context_tokens: int = 1400
    maximum_prompt_tokens: int = 2048
    reserved_answer_tokens: int = 400
    tokenizer_id: str = "word-counter-v1"
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    temperature: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_context_tokens < 1:
            raise ValueError("maximum_context_tokens must be positive.")
        if self.maximum_prompt_tokens < 1:
            raise ValueError("maximum_prompt_tokens must be positive.")
        if self.reserved_answer_tokens < 0:
            raise ValueError("reserved_answer_tokens must be >= 0.")
        if self.reserved_answer_tokens >= self.maximum_prompt_tokens:
            raise ValueError(
                "reserved_answer_tokens must be < maximum_prompt_tokens."
            )
        if self.maximum_context_tokens >= self.maximum_prompt_tokens:
            raise ValueError(
                "maximum_context_tokens must be < maximum_prompt_tokens "
                "(instruction and question share the same budget)."
            )


def answer(
    *,
    query_id: str,
    query_text: str,
    dataset_id: str,
    retrieved_chunks: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
    runtime: GenerationRuntimeAdapter | None = None,
    config: PipelineConfig | None = None,
    token_counter: TokenCounter | None = None,
    metadata: dict[str, Any] | None = None,
) -> GeneratedAnswer:
    """Run the full RAG pipeline: chunks -> context -> prompt -> answer.

    Stages:
        1. build_context_pack   (token-budgeted, deduplicated)
        2. build_generation_prompt  (validates citation integrity)
        3. runtime.generate     (QwenLLM by default)
        4. build_generated_answer  (validates answer citations)

    Args:
        query_id, query_text, dataset_id:
            Identity of the query being answered.
        retrieved_chunks:
            Chunks returned by retrieve_hybrid_rrf / retrieve_dense /
            retrieve_sparse. Must all belong to dataset_id.
        runtime:
            GenerationRuntimeAdapter. Defaults to QwenLLMAdapter()
            which lazily loads Qwen2.5-14B-Instruct.
        config:
            PipelineConfig with token budgets. Defaults are safe for
            the QwenLLM 2048-token hardcoded ceiling.
        token_counter:
            Callable[[str], int]. Defaults to word count.
        metadata:
            Optional extra metadata threaded into both context_pack
            and answer artifacts.

    Returns:
        GeneratedAnswer with validated citations.
    """

    cfg = config or PipelineConfig()
    runtime = runtime or QwenLLMAdapter()
    counter = token_counter or _default_word_counter

    if not query_id.strip():
        raise ValueError("query_id cannot be empty.")
    if not query_text.strip():
        raise ValueError("query_text cannot be empty.")
    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be empty.")
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks cannot be empty.")

    chunk_datasets = {c.dataset for c in retrieved_chunks}
    if len(chunk_datasets) > 1:
        raise ValueError(
            "All chunks must belong to the same dataset. "
            f"Got: {sorted(chunk_datasets)}"
        )
    if chunk_datasets and dataset_id not in chunk_datasets:
        raise ValueError(
            f"dataset_id={dataset_id!r} does not match chunk "
            f"dataset {chunk_datasets.pop()!r}."
        )

    retrieval_method = retrieved_chunks[0].retrieval_method

    context_pack = build_context_pack(
        query_id=query_id,
        query_text=query_text,
        dataset_id=dataset_id,
        retrieval_method=retrieval_method,
        tokenizer_id=cfg.tokenizer_id,
        maximum_context_tokens=cfg.maximum_context_tokens,
        retrieved_chunks=retrieved_chunks,
        token_counter=counter,
        metadata=metadata,
    )

    if context_pack.selected_item_count == 0:
        raise ValueError(
            "No chunks fit within the token budget "
            f"(maximum_context_tokens={cfg.maximum_context_tokens}). "
            "Increase the budget or provide shorter chunks. "
            f"All {len(retrieved_chunks)} chunks were skipped."
        )

    prompt = build_generation_prompt(
        context_pack=context_pack,
        token_counter=counter,
        maximum_prompt_tokens=cfg.maximum_prompt_tokens,
        reserved_answer_tokens=cfg.reserved_answer_tokens,
    )

    request = GenerationRequest(
        prompt=prompt,
        model_id=cfg.model_id,
        maximum_output_tokens=cfg.reserved_answer_tokens,
        temperature=cfg.temperature,
        seed=cfg.seed,
        response_format="text",
    )

    response = runtime.generate(request)

    return build_generated_answer(
        request=request,
        response=response,
        metadata=metadata,
    )


__all__ = [
    "PipelineConfig",
    "answer",
]