"""Deterministic identities for retrieval evaluations."""

from __future__ import annotations

from sah_rag.artifacts import (
    normalize_identifier,
    short_fingerprint,
)


def build_evaluation_run_id(
    *,
    dataset_id: str,
    evaluation_level: str,
    retrieval_methods: tuple[str, ...],
    cutoffs: tuple[int, ...],
    configuration_hash: str,
) -> str:
    """Build a deterministic retrieval-evaluation run ID."""
    identity = {
        "dataset_id": dataset_id,
        "evaluation_level": evaluation_level,
        "retrieval_methods": retrieval_methods,
        "cutoffs": cutoffs,
        "configuration_hash": configuration_hash,
    }

    return (
        f"evaluation-"
        f"{normalize_identifier(dataset_id)}-"
        f"{normalize_identifier(evaluation_level)}-"
        f"{short_fingerprint(identity)}"
    )
