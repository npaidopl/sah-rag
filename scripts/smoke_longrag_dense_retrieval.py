"""Synthetic end-to-end smoke for dense LongRAG retrieval.

Uses in-memory Qdrant and deterministic one-hot vectors. No foreign
LongRAG code is executed, no BGE-M3 inference is performed.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from rag_system.baselines import (
    LongUnitEmbeddingRecord,
    build_longrag_point_id,
    create_longrag_collection,
    expand_ranked_long_units_to_documents,
    ingest_longrag_records,
    retrieve_long_units,
    retrieved_to_ranked_long_units,
)

DIMENSION = 8
CONDITION = "LONGRAG-SMOKE-4096"
CONFIG_HASH = "smoke-config-hash"
DATASET = "synthetic-longrag"
COLLECTION = "smoke_longrag_dense"

UNITS: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("unit-1", ("doc-01", "doc-02"), 0),
    ("unit-2", ("doc-03", "doc-04"), 1),
    ("unit-3", ("doc-05", "doc-06"), 2),
    ("unit-4", ("doc-07", "doc-08"), 3),
    ("unit-5", ("doc-09", "doc-10"), 4),
    ("unit-6", ("doc-11", "doc-12"), 5),
)


def one_hot(index: int) -> tuple[float, ...]:
    return tuple(1.0 if i == index else 0.0 for i in range(DIMENSION))


def make_record(
    unit_id: str,
    doc_ids: tuple[str, ...],
    index: int,
) -> LongUnitEmbeddingRecord:
    return LongUnitEmbeddingRecord(
        long_unit_id=unit_id,
        condition_id=CONDITION,
        dataset_id=DATASET,
        construction_policy="adjacent_document_segments",
        target_token_size=4096,
        maximum_model_input_tokens=4098,
        unit_order=index,
        rendered_token_count=100 + index,
        special_token_count=2,
        actual_model_token_count=102 + index,
        was_truncated=False,
        member_doc_ids=doc_ids,
        member_parent_ids=tuple(f"parent-{d}" for d in doc_ids),
        member_chunk_ids=tuple(f"chunk-{d}" for d in doc_ids),
        model_id="BAAI/bge-m3",
        model_revision="smoke-rev",
        tokenizer_id="BAAI/bge-m3",
        embedding_config_hash=CONFIG_HASH,
        dense_vector=one_hot(index),
        dense_dimension=DIMENSION,
        dense_precision="float32",
        dense_normalized=True,
        source_unit_sha256="a" * 64,
        source_units_file_sha256="b" * 64,
        adapter_id="longrag_retrieval_adapter",
        adapter_version="0.2.0-source-corrected",
        source_architecture="LongRAG",
        source_architecture_commit="ba92f95",
    )


def main() -> None:
    records = [make_record(*u) for u in UNITS]

    client = QdrantClient(":memory:")

    print("[STEP] create collection")
    created = create_longrag_collection(
        client=client,
        collection_name=COLLECTION,
        dense_dimension=DIMENSION,
    )
    assert created is True
    assert client.collection_exists(COLLECTION)
    print("  created:", created)

    print("[STEP] first ingest")
    total1 = ingest_longrag_records(
        client=client,
        collection_name=COLLECTION,
        records=records,
    )
    count1 = client.count(COLLECTION).count
    print("  upserted:", total1)
    print("  count:", count1)
    assert total1 == 6
    assert count1 == 6

    print("[STEP] second ingest (idempotent)")
    total2 = ingest_longrag_records(
        client=client,
        collection_name=COLLECTION,
        records=records,
    )
    count2 = client.count(COLLECTION).count
    print("  upserted:", total2)
    print("  count:", count2)
    assert total2 == 6
    assert count2 == 6

    print("[STEP] deterministic point ids")
    pid_a = build_longrag_point_id(CONDITION, "unit-1", CONFIG_HASH)
    pid_b = build_longrag_point_id(CONDITION, "unit-1", CONFIG_HASH)
    assert pid_a == pid_b
    print("  sample:", pid_a)

    print("[STEP] queries")
    query_map = (
        ("q1", one_hot(0), "unit-1", "doc-01"),
        ("q2", one_hot(2), "unit-3", "doc-05"),
        ("q3", one_hot(5), "unit-6", "doc-11"),
    )
    for query_id, qvec, expected_unit, expected_doc in query_map:
        retrieved = retrieve_long_units(
            client=client,
            collection_name=COLLECTION,
            query_vector=qvec,
            limit=6,
        )
        assert len(retrieved) == 6
        assert retrieved[0].long_unit_id == expected_unit
        assert retrieved[0].score > 0.99

        ranked = retrieved_to_ranked_long_units(
            query_id=query_id,
            retrieved=retrieved,
        )
        expanded = expand_ranked_long_units_to_documents(
            query_id=query_id,
            ranked_long_units=ranked,
            document_limit=12,
        )
        assert expanded[0].doc_id == expected_doc

        print(
            f"  {query_id}: rank1={retrieved[0].long_unit_id} "
            f"score={retrieved[0].score:.4f} "
            f"top_doc={expanded[0].doc_id} "
            f"total_docs={len(expanded)}"
        )

    print("[STEP] payload round trip")
    retrieved = retrieve_long_units(
        client=client,
        collection_name=COLLECTION,
        query_vector=one_hot(0),
        limit=1,
    )
    assert retrieved[0].member_doc_ids == ("doc-01", "doc-02")
    assert retrieved[0].member_parent_ids == (
        "parent-doc-01",
        "parent-doc-02",
    )
    assert retrieved[0].payload["paper_faithful_result"] is False
    print("  memberships preserved:", retrieved[0].member_doc_ids)

    print("[STEP] condition filter")
    filtered = retrieve_long_units(
        client=client,
        collection_name=COLLECTION,
        query_vector=one_hot(0),
        limit=10,
        condition_id=CONDITION,
    )
    assert len(filtered) == 6
    print("  filtered count:", len(filtered))

    print("[STEP] cleanup")
    deleted = client.delete_collection(COLLECTION)
    assert deleted is True
    assert not client.collection_exists(COLLECTION)
    print("  deleted:", deleted)

    print("[OK] synthetic end-to-end smoke passed")


if __name__ == "__main__":
    main()
