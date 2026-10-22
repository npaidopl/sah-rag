"""Tests for embedding-shard writing and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sah_rag.artifacts import (
    ArtifactSidecarManifest,
    ChildChunk,
    OffsetSpan,
    read_json_model,
)
from sah_rag.artifacts.jsonl import (
    write_jsonl_atomic,
)
from sah_rag.embedding import (
    EmbeddingKind,
    EmbeddingShardHeader,
    VectorPrecision,
    create_synthetic_embedding,
    iter_embedding_records,
    write_embedding_shard,
)


def make_chunk(
    *,
    chunk_id: str,
) -> ChildChunk:
    """Create one minimal source chunk."""

    return ChildChunk(
        chunk_id=chunk_id,
        parent_id="parent-001",
        doc_id="doc-001",
        dataset="synthetic",
        chunk_index=0,
        parent_chunk_index=0,
        text_normalized="Example chunk.",
        text_raw="Example chunk.",
        normalized_char_span=OffsetSpan(
            start=0,
            end=14,
        ),
        parent_normalized_char_span=OffsetSpan(
            start=0,
            end=14,
        ),
        raw_char_span=OffsetSpan(
            start=0,
            end=14,
        ),
        token_span=OffsetSpan(
            start=0,
            end=3,
        ),
        token_count=3,
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


def make_header() -> EmbeddingShardHeader:
    """Create a hybrid embedding-shard header."""

    return EmbeddingShardHeader(
        shard_id="embedding-synthetic-000000-test",
        shard_index=0,
        dataset="synthetic",
        model_id="synthetic/model",
        model_revision="revision-001",
        tokenizer_id="tokenizer",
        embedding_config_hash="embedding-hash",
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=8,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
        maximum_records=10,
    )


def test_write_embedding_shard(
    tmp_path: Path,
) -> None:
    """A complete shard should publish all three files."""

    chunks = [
        make_chunk(chunk_id="chunk-001"),
        make_chunk(chunk_id="chunk-002"),
    ]

    source_path = tmp_path / "chunks.jsonl"

    write_jsonl_atomic(
        source_path,
        chunks,
    )

    header = make_header()

    records = [
        create_synthetic_embedding(
            chunk=chunk,
            model_id=header.model_id,
            model_revision=header.model_revision,
            tokenizer_id=header.tokenizer_id,
            embedding_config_hash=(header.embedding_config_hash),
            dense_dimension=8,
        )
        for chunk in chunks
    ]

    summary = write_embedding_shard(
        root=tmp_path / "shard-000000",
        header=header,
        records=records,
        source_chunks_path=source_path,
        execution_profile="test",
    )

    assert summary.record_count == 2
    assert summary.paths.header.is_file()
    assert summary.paths.records.is_file()
    assert summary.paths.manifest.is_file()

    loaded_records = list(iter_embedding_records(summary.paths.records))

    assert len(loaded_records) == 2

    manifest = read_json_model(
        summary.paths.manifest,
        ArtifactSidecarManifest,
    )

    assert manifest.record_count == 2
    assert len(manifest.inputs) == 2
    assert manifest.metadata["shard_id"] == (header.shard_id)

    assert manifest.metadata[
        "source_chunk_count"
    ] == 2

    assert manifest.metadata[
        "embedded_chunk_count"
    ] == 2

    assert manifest.metadata[
        "complete_source_coverage"
    ] is True


def test_duplicate_chunk_ids_are_rejected(
    tmp_path: Path,
) -> None:
    """One shard cannot contain duplicate chunk IDs."""

    chunk = make_chunk(chunk_id="chunk-001")
    source_path = tmp_path / "chunks.jsonl"

    write_jsonl_atomic(
        source_path,
        [chunk],
    )

    header = make_header()

    record = create_synthetic_embedding(
        chunk=chunk,
        model_id=header.model_id,
        model_revision=header.model_revision,
        tokenizer_id=header.tokenizer_id,
        embedding_config_hash=(header.embedding_config_hash),
        dense_dimension=8,
    )

    shard_root = tmp_path / "shard"

    with pytest.raises(
        ValueError,
        match="duplicate chunk_id",
    ):
        write_embedding_shard(
            root=shard_root,
            header=header,
            records=[record, record],
            source_chunks_path=source_path,
        )

    assert not (shard_root / "header.json").exists()

    assert not (shard_root / "records.jsonl").exists()

    assert not (shard_root / "shard.manifest.json").exists()


def test_record_header_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    """Record and shard model identities must agree."""

    chunk = make_chunk(chunk_id="chunk-001")
    source_path = tmp_path / "chunks.jsonl"

    write_jsonl_atomic(
        source_path,
        [chunk],
    )

    header = make_header()

    record = create_synthetic_embedding(
        chunk=chunk,
        model_id="different/model",
        model_revision=header.model_revision,
        tokenizer_id=header.tokenizer_id,
        embedding_config_hash=(header.embedding_config_hash),
        dense_dimension=8,
    )

    with pytest.raises(
        ValueError,
        match="model_id",
    ):
        write_embedding_shard(
            root=tmp_path / "shard",
            header=header,
            records=[record],
            source_chunks_path=source_path,
        )


def test_header_json_round_trip(
    tmp_path: Path,
) -> None:
    """The shard header should remain typed JSON."""

    header = make_header()
    path = tmp_path / "header.json"

    path.write_text(
        json.dumps(
            header.model_dump(mode="json"),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = read_json_model(
        path,
        EmbeddingShardHeader,
    )

    assert loaded == header


def test_unknown_source_chunk_is_rejected(
    tmp_path: Path,
) -> None:
    """Every embedding must reference a source chunk."""

    source_chunk = make_chunk(
        chunk_id="chunk-source"
    )

    embedded_chunk = make_chunk(
        chunk_id="chunk-unknown"
    )

    source_path = tmp_path / "chunks.jsonl"

    write_jsonl_atomic(
        source_path,
        [source_chunk],
    )

    header = make_header()

    record = create_synthetic_embedding(
        chunk=embedded_chunk,
        model_id=header.model_id,
        model_revision=header.model_revision,
        tokenizer_id=header.tokenizer_id,
        embedding_config_hash=(
            header.embedding_config_hash
        ),
        dense_dimension=8,
    )

    shard_root = tmp_path / "shard"

    with pytest.raises(
        ValueError,
        match="unknown source chunk_id",
    ):
        write_embedding_shard(
            root=shard_root,
            header=header,
            records=[record],
            source_chunks_path=source_path,
        )

    assert not (
        shard_root / "records.jsonl"
    ).exists()

    assert not (
        shard_root / "header.json"
    ).exists()

    assert not (
        shard_root / "shard.manifest.json"
    ).exists()
