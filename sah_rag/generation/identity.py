"""Deterministic identities for generation context packs."""

from __future__ import annotations

from sah_rag.artifacts import (
    normalize_identifier,
    short_fingerprint,
)


def build_context_pack_id(
    *,
    dataset_id: str,
    query_id: str,
    retrieval_method: str,
    tokenizer_id: str,
    maximum_context_tokens: int,
    selected_chunk_ids: tuple[str, ...],
) -> str:
    """Build a deterministic context-pack identifier."""

    identity = {
        "dataset_id": dataset_id,
        "query_id": query_id,
        "retrieval_method": retrieval_method,
        "tokenizer_id": tokenizer_id,
        "maximum_context_tokens": maximum_context_tokens,
        "selected_chunk_ids": selected_chunk_ids,
    }

    return (
        "context-"
        f"{normalize_identifier(dataset_id)}-"
        f"{normalize_identifier(query_id)}-"
        f"{short_fingerprint(identity)}"
    )
