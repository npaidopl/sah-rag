"""Tests for embedding-run catalog validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sah_rag.embedding import (
    EmbeddingRunCatalog,
    EmbeddingShardCatalogEntry,
)


def make_entry(
    *,
    index: int,
    start: int,
    end: int,
) -> EmbeddingShardCatalogEntry:
    """Create one catalog entry."""

    digest = "a" * 64

    return EmbeddingShardCatalogEntry(
        shard_id=f"shard-{index}",
        shard_index=index,
        shard_root=f"/tmp/shard-{index}",
        header_path=f"/tmp/shard-{index}/header.json",
        records_path=f"/tmp/shard-{index}/records.jsonl",
        manifest_path=f"/tmp/shard-{index}/manifest.json",
        header_sha256=digest,
        records_sha256=digest,
        manifest_sha256=digest,
        record_count=end - start,
        source_start_index=start,
        source_end_index=end,
    )


def make_catalog(
    entries: tuple[
        EmbeddingShardCatalogEntry,
        ...
    ],
) -> EmbeddingRunCatalog:
    """Create one test catalog."""

    return EmbeddingRunCatalog(
        run_id="run-001",
        dataset_id="synthetic",
        model_id="test/model",
        model_revision="revision",
        tokenizer_id="tokenizer",
        embedding_config_hash="b" * 64,
        embedding_kind="hybrid",
        dense_dimension=3,
        dense_precision="float32",
        dense_normalized=True,
        source_chunks_path="/tmp/chunks.jsonl",
        source_chunks_manifest_path="/tmp/chunks.manifest.json",
        source_chunks_sha256="c" * 64,
        shard_count=len(entries),
        record_count=sum(
            entry.record_count
            for entry in entries
        ),
        document_count=2,
        shards=entries,
    )


def test_catalog_accepts_contiguous_ranges() -> None:
    """Contiguous shard ranges should validate."""

    catalog = make_catalog(
        (
            make_entry(
                index=0,
                start=0,
                end=2,
            ),
            make_entry(
                index=1,
                start=2,
                end=5,
            ),
        )
    )

    assert catalog.record_count == 5


def test_catalog_rejects_range_gap() -> None:
    """Embedding shard ranges cannot contain gaps."""

    with pytest.raises(
        ValidationError,
        match="not contiguous",
    ):
        make_catalog(
            (
                make_entry(
                    index=0,
                    start=0,
                    end=2,
                ),
                make_entry(
                    index=1,
                    start=3,
                    end=5,
                ),
            )
        )


def test_catalog_rejects_non_sha256_hash() -> None:
    """Embedding configuration identity must be SHA-256."""

    catalog = make_catalog(
        (
            make_entry(
                index=0,
                start=0,
                end=2,
            ),
        )
    )

    with pytest.raises(ValidationError):
        catalog.model_copy(
            update={
                "embedding_config_hash": "label",
            }
        ).__class__.model_validate(
            {
                **catalog.model_dump(
                    mode="json"
                ),
                "embedding_config_hash": "label",
            }
        )
