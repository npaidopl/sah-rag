"""Tests for LongRAG dense Qdrant storage helpers."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from baselines import (
    LongUnitEmbeddingRecord,
    build_longrag_point,
    build_longrag_point_id,
    create_longrag_collection,
    fingerprint_longrag_collection_name,
    ingest_longrag_records,
)


def make_record(
    *,
    long_unit_id: str = "longunit-clerk-a",
    condition_id: str = "LONGRAG-SD-4096",
    config_hash: str = "cfg-abc",
    dim: int = 4,
    members: tuple[str, ...] = ("doc-a", "doc-b"),
) -> LongUnitEmbeddingRecord:
    return LongUnitEmbeddingRecord(
        long_unit_id=long_unit_id,
        condition_id=condition_id,
        dataset_id="clerk",
        construction_policy="single_document_if_within_budget",
        target_token_size=4096,
        maximum_model_input_tokens=1024,
        unit_order=0,
        rendered_token_count=100,
        special_token_count=2,
        actual_model_token_count=102,
        was_truncated=False,
        member_doc_ids=members,
        member_parent_ids=tuple(f"parent-{m}" for m in members),
        member_chunk_ids=tuple(f"chunk-{m}" for m in members),
        model_id="BAAI/bge-m3",
        model_revision="rev",
        tokenizer_id="BAAI/bge-m3",
        embedding_config_hash=config_hash,
        dense_vector=tuple([1.0] + [0.0] * (dim - 1)),
        dense_dimension=dim,
        dense_precision="float32",
        dense_normalized=True,
        source_unit_sha256="s" * 64,
        source_units_file_sha256="f" * 64,
        adapter_id="longrag_retrieval_adapter",
        adapter_version="0.2.0-source-corrected",
        source_architecture="LongRAG",
        source_architecture_commit="ba92f95",
    )


def test_point_id_is_deterministic_and_unique() -> None:
    a = build_longrag_point_id("C", "U", "H")
    b = build_longrag_point_id("C", "U", "H")
    c = build_longrag_point_id("C", "U", "H2")
    d = build_longrag_point_id("C2", "U", "H")
    assert a == b
    assert a != c
    assert a != d


def test_build_point_payload_round_trip() -> None:
    record = make_record()
    point = build_longrag_point(record)
    assert point.payload is not None
    assert point.payload["long_unit_id"] == "longunit-clerk-a"
    assert point.payload["member_doc_ids"] == ["doc-a", "doc-b"]
    assert point.payload["paper_faithful_result"] is False
    assert point.payload["interface_validation"] is True


def test_create_collection_and_idempotent_ingest() -> None:
    client = QdrantClient(":memory:")
    name = "test_longrag"

    created = create_longrag_collection(
        client=client,
        collection_name=name,
        dense_dimension=4,
    )
    assert created is True
    assert client.collection_exists(name)

    records = [make_record(long_unit_id=f"u{i}") for i in range(3)]

    total1 = ingest_longrag_records(
        client=client,
        collection_name=name,
        records=records,
    )
    assert total1 == 3
    assert client.count(name).count == 3

    total2 = ingest_longrag_records(
        client=client,
        collection_name=name,
        records=records,
    )
    assert total2 == 3
    assert client.count(name).count == 3

    client.delete_collection(name)
    assert not client.collection_exists(name)


def test_fingerprint_collection_name_is_deterministic() -> None:
    a = fingerprint_longrag_collection_name(
        dataset_id="clerk",
        condition_id="LONGRAG-SD-4096",
        embedding_config_hash="x",
    )
    b = fingerprint_longrag_collection_name(
        dataset_id="clerk",
        condition_id="LONGRAG-SD-4096",
        embedding_config_hash="x",
    )
    c = fingerprint_longrag_collection_name(
        dataset_id="clerk",
        condition_id="LONGRAG-SD-4096",
        embedding_config_hash="y",
    )
    assert a == b
    assert a != c
    assert a.startswith("clerk_longrag_longrag-sd-4096_")


def test_duplicate_long_unit_id_in_batch_is_rejected() -> None:
    client = QdrantClient(":memory:")
    create_longrag_collection(
        client=client,
        collection_name="dup",
        dense_dimension=4,
    )
    record = make_record()
    with pytest.raises(ValueError, match="Duplicate point id"):
        ingest_longrag_records(
            client=client,
            collection_name="dup",
            records=[record, record],
        )
