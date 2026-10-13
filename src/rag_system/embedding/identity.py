"""Deterministic identities for embedding shards."""

from __future__ import annotations

from rag_system.artifacts.identity import (
    normalize_identifier,
    short_fingerprint,
)


def build_embedding_shard_id(
    *,
    dataset: str,
    model_id: str,
    embedding_config_hash: str,
    shard_index: int,
) -> str:
    """Build a deterministic embedding-shard ID."""

    if shard_index < 0:
        raise ValueError("shard_index cannot be negative.")

    identity = {
        "dataset": dataset,
        "model_id": model_id,
        "embedding_config_hash": (embedding_config_hash),
        "shard_index": shard_index,
    }

    suffix = short_fingerprint(identity)

    return f"embedding-{normalize_identifier(dataset)}-{shard_index:06d}-{suffix}"
