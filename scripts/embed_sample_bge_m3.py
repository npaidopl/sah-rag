"""Embed the complete BGE-M3 sample chunk artifact."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from FlagEmbedding import BGEM3FlagModel

from rag_system.artifacts import (
    ChildChunk,
    sha256_fingerprint,
)
from rag_system.canonical.jsonl import iter_jsonl
from rag_system.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingKind,
    EmbeddingShardHeader,
    SparseVector,
    VectorPrecision,
    build_embedding_shard_id,
    write_embedding_shard,
)

CHUNKS_PATH = Path("outputs/clerk_chunks_auto.jsonl")
CHUNKS_MANIFEST_PATH = Path("outputs/clerk_chunks_auto.manifest.json")
MODEL_REPORT_PATH = Path("outputs/bge_m3_model_download.json")
SHARD_ROOT = Path("outputs/bge_m3_sample_shard_000000")

BATCH_SIZE = 1
MAXIMUM_MODEL_INPUT_TOKENS = 256
DENSE_DIMENSION = 1024


def convert_sparse_vector(
    lexical_weights: dict[Any, Any],
) -> SparseVector:
    """Convert BGE lexical weights into sorted sparse arrays."""
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


def dense_norm(vector: tuple[float, ...]) -> float:
    """Calculate a dense-vector L2 norm."""
    return math.sqrt(sum(value * value for value in vector))


def main() -> None:
    """Embed all sample chunks with the pinned BGE-M3 model."""
    model_report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    model_id = str(model_report["model_id"])
    model_revision = str(model_report["revision"])
    snapshot_path = str(model_report["snapshot_path"])

    chunks = [
        ChildChunk.model_validate(record)
        for record in iter_jsonl(CHUNKS_PATH)
    ]

    if not chunks:
        raise RuntimeError("The source chunk artifact is empty.")

    dataset_ids = {chunk.dataset for chunk in chunks}
    if len(dataset_ids) != 1:
        raise RuntimeError("The sample contains multiple datasets.")

    dataset_id = next(iter(dataset_ids))
    tokenizer_id = f"huggingface:{model_id}@{model_revision}:fast"

    configuration = {
        "embedder_id": "bge_m3_embedder",
        "embedder_version": "0.1.0",
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "embedding_kind": "hybrid",
        "dense_dimension": DENSE_DIMENSION,
        "dense_precision": "float32",
        "dense_normalized": True,
        "device": "cpu",
        "use_fp16": False,
        "batch_size": BATCH_SIZE,
        "maximum_model_input_tokens": MAXIMUM_MODEL_INPUT_TOKENS,
        "source_chunk_count": len(chunks),
    }

    config_hash = sha256_fingerprint(configuration)

    print("[INFO] Loading model:", model_id)
    print("[INFO] Revision:", model_revision)
    print("[INFO] Snapshot:", snapshot_path)
    print("[INFO] Device: cpu")
    print("[INFO] Batch size:", BATCH_SIZE)
    print("[INFO] Source chunks:", len(chunks))

    load_started = time.perf_counter()
    model = BGEM3FlagModel(snapshot_path, use_fp16=False, device="cpu")
    load_seconds = time.perf_counter() - load_started

    texts = [chunk.text_normalized for chunk in chunks]
    encode_started = time.perf_counter()
    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAXIMUM_MODEL_INPUT_TOKENS,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    encode_seconds = time.perf_counter() - encode_started

    dense_vectors = output["dense_vecs"]
    lexical_weights = output["lexical_weights"]

    if len(dense_vectors) != len(chunks):
        raise RuntimeError("Dense output count does not match the source chunk count.")
    if len(lexical_weights) != len(chunks):
        raise RuntimeError("Sparse output count does not match the source chunk count.")

    records: list[ChunkEmbeddingRecord] = []
    for index, chunk in enumerate(chunks):
        dense_vector = tuple(float(value) for value in dense_vectors[index])
        if len(dense_vector) != DENSE_DIMENSION:
            raise RuntimeError(f"Unexpected dense-vector dimension: {len(dense_vector)}")

        norm = dense_norm(dense_vector)
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError(f"Dense vector is not normalized: {norm}")

        sparse_vector = convert_sparse_vector(lexical_weights[index])
        records.append(
            ChunkEmbeddingRecord(
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                doc_id=chunk.doc_id,
                dataset=chunk.dataset,
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_id=tokenizer_id,
                embedding_config_hash=config_hash,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                dense_dimension=DENSE_DIMENSION,
                dense_precision=VectorPrecision.FLOAT32,
                dense_normalized=True,
            )
        )

    shard_id = build_embedding_shard_id(
        dataset=dataset_id,
        model_id=model_id,
        embedding_config_hash=config_hash,
        shard_index=0,
    )

    header = EmbeddingShardHeader(
        shard_id=shard_id,
        shard_index=0,
        dataset=dataset_id,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        embedding_config_hash=config_hash,
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=DENSE_DIMENSION,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
        maximum_records=len(chunks),
    )

    summary = write_embedding_shard(
        root=SHARD_ROOT,
        header=header,
        records=records,
        source_chunks_path=CHUNKS_PATH,
        source_chunks_manifest_path=CHUNKS_MANIFEST_PATH,
        overwrite=True,
        execution_profile="local_cpu",
        metadata={
            "synthetic": False,
            "semantic_embeddings": True,
            "smoke_test": False,
            "sample_run": True,
            "model_load_seconds": round(load_seconds, 3),
            "encode_seconds": round(encode_seconds, 3),
            "batch_size": BATCH_SIZE,
            "maximum_model_input_tokens": MAXIMUM_MODEL_INPUT_TOKENS,
        },
    )

    sparse_counts = [len(r.sparse_vector.indices) for r in records if r.sparse_vector is not None]
    dense_norms = [dense_norm(r.dense_vector) for r in records if r.dense_vector is not None]

    print("[OK] BGE-M3 sample shard")
    print(" Shard ID:", summary.shard_id)
    print(" Records:", summary.record_count)
    print(" Dense dimension:", DENSE_DIMENSION)
    print(" Dense norm range:", round(min(dense_norms), 6), round(max(dense_norms), 6))
    print(" Sparse count range:", min(sparse_counts), max(sparse_counts))
    print(" Model load seconds:", round(load_seconds, 3))
    print(" Encode seconds:", round(encode_seconds, 3))
    print(" Records file:", summary.paths.records.resolve())
    print(" Manifest:", summary.paths.manifest.resolve())


if __name__ == "__main__":
    main()
