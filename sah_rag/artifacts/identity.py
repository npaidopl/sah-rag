"""Deterministic fingerprints and identifiers for RAG artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_IDENTIFIER_PATTERN = re.compile(r"[^a-z0-9]+")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return serialized.encode("utf-8")


def sha256_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint."""

    return hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()


def short_fingerprint(
    value: Any,
    *,
    length: int = 12,
) -> str:
    """Return a shortened deterministic fingerprint."""

    if length < 8 or length > 64:
        raise ValueError(
            "Fingerprint length must be between 8 and 64."
        )

    return sha256_fingerprint(value)[:length]


def normalize_identifier(value: str) -> str:
    """Normalize a human-readable value for stable identifiers."""

    normalized = _IDENTIFIER_PATTERN.sub(
        "-",
        value.lower().strip(),
    ).strip("-")

    if not normalized:
        raise ValueError(
            "Identifier must contain at least one letter or number."
        )

    return normalized


def build_parent_id(
    *,
    dataset: str,
    doc_id: str,
    parser_config_hash: str,
    parent_index: int,
) -> str:
    """Build a deterministic semantic-parent identifier."""

    if parent_index < 0:
        raise ValueError("parent_index cannot be negative.")

    identity = {
        "dataset": dataset,
        "doc_id": doc_id,
        "parser_config_hash": parser_config_hash,
        "parent_index": parent_index,
    }

    suffix = short_fingerprint(identity)
    return (
        f"parent-{normalize_identifier(dataset)}-"
        f"{parent_index:06d}-{suffix}"
    )


def build_chunk_id(
    *,
    dataset: str,
    doc_id: str,
    parent_id: str,
    chunker_config_hash: str,
    parent_chunk_index: int,
) -> str:
    """Build a deterministic child-chunk identifier."""

    if parent_chunk_index < 0:
        raise ValueError(
            "parent_chunk_index cannot be negative."
        )

    identity = {
        "dataset": dataset,
        "doc_id": doc_id,
        "parent_id": parent_id,
        "chunker_config_hash": chunker_config_hash,
        "parent_chunk_index": parent_chunk_index,
    }

    suffix = short_fingerprint(identity)

    return (
        f"chunk-{normalize_identifier(dataset)}-"
        f"{parent_chunk_index:06d}-{suffix}"
    )


def build_experiment_id(
    *,
    architecture_id: str,
    dataset_id: str,
    configuration: Mapping[str, Any],
) -> str:
    """Build a deterministic experiment identifier."""

    architecture = normalize_identifier(architecture_id)
    dataset = normalize_identifier(dataset_id)
    fingerprint = short_fingerprint(
        {
            "architecture_id": architecture,
            "dataset_id": dataset,
            "configuration": dict(configuration),
        }
    )

    return f"{architecture}__{dataset}__{fingerprint}"


def combine_fingerprints(
    fingerprints: Sequence[str],
) -> str:
    """Combine ordered component fingerprints."""

    if not fingerprints:
        raise ValueError(
            "At least one fingerprint is required."
        )

    if any(not value.strip() for value in fingerprints):
        raise ValueError(
            "Fingerprints cannot contain empty values."
        )

    return sha256_fingerprint(list(fingerprints))