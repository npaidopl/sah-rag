"""Deterministic context selection under a token budget."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from sah_rag.generation.identity import (
    build_context_pack_id,
)
from sah_rag.generation.models import (
    ContextItem,
    ContextPack,
)
from sah_rag.retrieval import RetrievedChunk

TokenCounter = Callable[[str], int]


def build_context_pack(
    *,
    query_id: str,
    query_text: str,
    dataset_id: str,
    retrieval_method: str,
    tokenizer_id: str,
    maximum_context_tokens: int,
    retrieved_chunks: Sequence[RetrievedChunk],
    token_counter: TokenCounter,
    maximum_chunks_per_document: int | None = None,
    skip_oversized_items: bool = True,
    metadata: dict[str, object] | None = None,
) -> ContextPack:
    """Select ranked chunks without exceeding a token budget."""

    required_values = (
        query_id,
        query_text,
        dataset_id,
        retrieval_method,
        tokenizer_id,
    )

    if any(not value.strip() for value in required_values):
        raise ValueError("Context-builder identity fields cannot be empty.")

    if maximum_context_tokens < 1:
        raise ValueError("maximum_context_tokens must be positive.")

    if maximum_chunks_per_document is not None and maximum_chunks_per_document < 1:
        raise ValueError("maximum_chunks_per_document must be positive.")

    ordered_chunks = sorted(
        retrieved_chunks,
        key=lambda chunk: (
            chunk.rank,
            -chunk.score,
            chunk.chunk_id,
        ),
    )

    seen_chunk_ids: set[str] = set()
    document_counts: dict[str, int] = {}

    selected: list[tuple[RetrievedChunk, int]] = []
    skipped_chunk_ids: list[str] = []

    used_tokens = 0

    for chunk in ordered_chunks:
        if chunk.chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk.chunk_id)

        document_count = document_counts.get(chunk.doc_id, 0)

        if (
            maximum_chunks_per_document is not None
            and document_count >= maximum_chunks_per_document
        ):
            skipped_chunk_ids.append(chunk.chunk_id)
            continue

        token_count = token_counter(chunk.text_normalized)

        if token_count < 1:
            raise ValueError("Context token counter returned a non-positive value.")

        fits = used_tokens + token_count <= maximum_context_tokens

        if not fits:
            skipped_chunk_ids.append(chunk.chunk_id)

            if skip_oversized_items:
                continue

            break

        selected.append((chunk, token_count))
        used_tokens += token_count
        document_counts[chunk.doc_id] = document_count + 1

    selected_items = tuple(
        ContextItem(
            citation_id=f"[C{index}]",
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
            doc_id=chunk.doc_id,
            dataset=chunk.dataset,
            retrieval_rank=chunk.rank,
            retrieval_score=chunk.score,
            retrieval_method=chunk.retrieval_method,
            text=chunk.text_normalized,
            token_count=token_count,
            metadata={"source_payload": chunk.payload},
        )
        for index, (chunk, token_count) in enumerate(selected, start=1)
    )

    selected_chunk_ids = tuple(item.chunk_id for item in selected_items)

    context_pack_id = build_context_pack_id(
        dataset_id=dataset_id,
        query_id=query_id,
        retrieval_method=retrieval_method,
        tokenizer_id=tokenizer_id,
        maximum_context_tokens=maximum_context_tokens,
        selected_chunk_ids=selected_chunk_ids,
    )

    return ContextPack(
        context_pack_id=context_pack_id,
        query_id=query_id,
        query_text=query_text,
        dataset_id=dataset_id,
        retrieval_method=retrieval_method,
        tokenizer_id=tokenizer_id,
        maximum_context_tokens=maximum_context_tokens,
        used_context_tokens=used_tokens,
        remaining_context_tokens=maximum_context_tokens - used_tokens,
        selected_item_count=len(selected_items),
        skipped_item_count=len(skipped_chunk_ids),
        selected_items=selected_items,
        skipped_chunk_ids=tuple(skipped_chunk_ids),
        metadata=metadata if metadata is not None else {},
    )
