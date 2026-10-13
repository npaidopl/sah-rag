"""Embed a real query and run dense, sparse, and RRF retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient

from rag_system.artifacts import (
    ChildChunk,
    read_json_model,
)
from rag_system.canonical.jsonl import iter_jsonl
from rag_system.embedding import (
    EmbeddingShardHeader,
    SparseVector,
    iter_embedding_records,
)
from rag_system.retrieval import (
    retrieve_dense,
    retrieve_hybrid_parents,
    retrieve_hybrid_rrf,
    retrieve_sparse,
)
from rag_system.storage import (
    create_hybrid_collection,
    fingerprint_collection_name,
    ingest_embedding_records,
)

QUERY = (
    "electrical appliance casing and supporting plate"
)

CHUNKS_PATH = Path(
    "outputs/clerk_chunks_auto.jsonl"
)

SHARD_ROOT = Path(
    "outputs/bge_m3_sample_shard_000000"
)

MODEL_REPORT_PATH = Path(
    "outputs/bge_m3_model_download.json"
)


def convert_sparse(
    lexical_weights: dict[Any, Any],
) -> SparseVector:
    """Convert query lexical weights to a sparse vector."""
    items = sorted(
        (
            int(index),
            float(value),
        )
        for index, value in lexical_weights.items()
        if float(value) != 0.0
    )

    return SparseVector(
        indices=tuple(index for index, _ in items),
        values=tuple(value for _, value in items),
    )


def print_chunks(
    label: str,
    results: tuple[object, ...],
) -> None:
    """Print compact ranked chunk results."""
    print(label)

    for result in results:
        print(
            " ",
            result.rank,
            result.chunk_id,
            "score=",
            round(result.score, 6),
            "parent=",
            result.parent_id,
        )


def main() -> None:
    """Run real BGE-M3 query retrieval over eight chunks."""
    model_report = json.loads(
        MODEL_REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    header = read_json_model(
        SHARD_ROOT / "header.json",
        EmbeddingShardHeader,
    )

    records = list(
        iter_embedding_records(
            SHARD_ROOT / "records.jsonl"
        )
    )

    chunks = [
        ChildChunk.model_validate(record)
        for record in iter_jsonl(CHUNKS_PATH)
    ]

    collection_name = fingerprint_collection_name(
        dataset=header.dataset,
        embedding_config_hash=(
            header.embedding_config_hash
        ),
    )

    client = QdrantClient(":memory:")

    create_hybrid_collection(
        client=client,
        collection_name=collection_name,
        dense_dimension=(
            header.dense_dimension or 0
        ),
    )

    ingest_embedding_records(
        client=client,
        collection_name=collection_name,
        header=header,
        records=records,
        chunks=chunks,
        batch_size=4,
    )

    print("[INFO] Query:", QUERY)
    print("[INFO] Loading model.")

    model = BGEM3FlagModel(
        model_report["snapshot_path"],
        use_fp16=False,
        device="cpu",
    )

    started = time.perf_counter()

    output = model.encode(
        [QUERY],
        batch_size=1,
        max_length=256,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    elapsed = time.perf_counter() - started

    dense_vector = tuple(
        float(value)
        for value in output["dense_vecs"][0]
    )

    sparse_vector = convert_sparse(
        output["lexical_weights"][0]
    )

    dense_results = retrieve_dense(
        client=client,
        collection_name=collection_name,
        dense_vector=dense_vector,
        limit=5,
    )

    sparse_results = retrieve_sparse(
        client=client,
        collection_name=collection_name,
        sparse_indices=sparse_vector.indices,
        sparse_values=sparse_vector.values,
        limit=5,
    )

    hybrid_results = retrieve_hybrid_rrf(
        client=client,
        collection_name=collection_name,
        dense_vector=dense_vector,
        sparse_indices=sparse_vector.indices,
        sparse_values=sparse_vector.values,
        candidate_limit=8,
        result_limit=5,
    )

    parent_results = retrieve_hybrid_parents(
        client=client,
        collection_name=collection_name,
        dense_vector=dense_vector,
        sparse_indices=sparse_vector.indices,
        sparse_values=sparse_vector.values,
        candidate_limit=8,
        child_limit=8,
        parent_limit=5,
    )

    print("[OK] Query embedding seconds:", round(elapsed, 3))
    print(
        "[OK] Query sparse non-zero:",
        len(sparse_vector.indices),
    )

    print_chunks("[DENSE]", dense_results)
    print_chunks("[SPARSE]", sparse_results)
    print_chunks("[HYBRID RRF]", hybrid_results)

    print("[PARENTS]")

    for parent in parent_results:
        print(
            " ",
            parent.rank,
            parent.parent_id,
            "score=",
            round(parent.score, 6),
            "children=",
            parent.matched_chunk_count,
            "best_child_rank=",
            parent.best_chunk_rank,
        )


if __name__ == "__main__":
    main()
