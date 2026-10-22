"""Tests for deterministic synthetic embeddings."""

from __future__ import annotations

import math

from sah_rag.artifacts import ChildChunk, OffsetSpan
from sah_rag.embedding import (
    create_synthetic_embedding,
)


def make_chunk() -> ChildChunk:
    """Create one source chunk."""

    return ChildChunk(
        chunk_id="chunk-001",
        parent_id="parent-001",
        doc_id="doc-001",
        dataset="synthetic",
        chunk_index=0,
        parent_chunk_index=0,
        text_normalized="Example.",
        text_raw="Example.",
        normalized_char_span=OffsetSpan(
            start=0,
            end=8,
        ),
        parent_normalized_char_span=OffsetSpan(
            start=0,
            end=8,
        ),
        raw_char_span=OffsetSpan(
            start=0,
            end=8,
        ),
        token_span=OffsetSpan(
            start=0,
            end=2,
        ),
        token_count=2,
        parser_id="parser",
        parser_version="0.3.0",
        parser_config_hash="parser-hash",
        chunker_id="chunker",
        chunker_version="0.3.0",
        chunker_config_hash="chunker-hash",
        tokenizer_id="tokenizer",
        chunk_size=256,
        chunk_overlap=32,
        metadata={},
    )


def test_synthetic_embedding_is_deterministic() -> None:
    """Equivalent inputs should create equal records."""

    chunk = make_chunk()

    kwargs = {
        "chunk": chunk,
        "model_id": "synthetic/model",
        "model_revision": "revision-001",
        "tokenizer_id": "tokenizer",
        "embedding_config_hash": "hash",
        "dense_dimension": 8,
    }

    first = create_synthetic_embedding(**kwargs)
    second = create_synthetic_embedding(**kwargs)

    assert first == second


def test_synthetic_dense_vector_is_normalized() -> None:
    """Synthetic dense vectors should have unit norm."""

    record = create_synthetic_embedding(
        chunk=make_chunk(),
        model_id="synthetic/model",
        model_revision="revision-001",
        tokenizer_id="tokenizer",
        embedding_config_hash="hash",
        dense_dimension=16,
    )

    assert record.dense_vector is not None

    norm = math.sqrt(sum(value * value for value in record.dense_vector))

    assert math.isclose(
        norm,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
