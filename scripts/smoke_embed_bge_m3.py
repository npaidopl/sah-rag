"""Create a real BGE-M3 embedding shard for two chunks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from FlagEmbedding import BGEM3FlagModel

from sah_rag.artifacts import (
    ChildChunk,
    sha256_fingerprint,
)
from sah_rag.canonical.jsonl import iter_jsonl
from sah_rag.embedding import (
    ChunkEmbeddingRecord,
    EmbeddingKind,
    EmbeddingShardHeader,
    SparseVector,
    VectorPrecision,
    build_embedding_shard_id,
    write_embedding_shard,
)

CHUNKS_PATH = Path(
    "outputs/clerk_chunks_auto.jsonl"
)

CHUNKS_MANIFEST_PATH = Path(
    "outputs/clerk_chunks_auto.manifest.json"
)

MODEL_REPORT_PATH = Path(
    "outputs/bge_m3_model_download.json"
)

SHARD_ROOT = Path(
    "outputs/bge_m3_smoke_shard_000000"
)

SMOKE_LIMIT = 2
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
        indices=tuple(
            index
            for index, _ in items
        ),
        values=tuple(
            value
            for _, value in items
        ),
    )


def main() -> None:
    """Load the pinned model and embed two sample chunks."""

    model_report = json.loads(
        MODEL_REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    model_id = model_report["model_id"]
    model_revision = model_report["revision"]
    snapshot_path = model_report["snapshot_path"]

    chunks = [
        ChildChunk.model_validate(record)
        for record in iter_jsonl(CHUNKS_PATH)
    ][:SMOKE_LIMIT]

    if len(chunks) != SMOKE_LIMIT:
        raise RuntimeError(
            "The smoke test requires exactly two chunks."
        )

    configuration = {
        "embedder_id": "bge_m3_embedder",
        "embedder_version": "0.1.0",
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": (
            f"huggingface:{model_id}"
            f"@{model_revision}:fast"
        ),
        "embedding_kind": "hybrid",
        "dense_dimension": DENSE_DIMENSION,
        "dense_precision": "float32",
        "dense_normalized": True,
        "device": "cpu",
        "use_fp16": False,
        "batch_size": 1,
        "maximum_model_input_tokens": 256,
        "smoke_limit": SMOKE_LIMIT,
    }

    config_hash = sha256_fingerprint(
        configuration
    )

    print("[INFO] Loading model from:", snapshot_path)
    print("[INFO] Revision:", model_revision)
    print("[INFO] Device: cpu")
    print("[INFO] FP16: False")

    started = time.perf_counter()

    model = BGEM3FlagModel(
        snapshot_path,
        use_fp16=False,
        device="cpu",
    )

    load_seconds = time.perf_counter() - started

    texts = [
        chunk.text_normalized
        for chunk in chunks
    ]

    encode_started = time.perf_counter()

    output = model.encode(
        texts,
        batch_size=1,
        max_length=256,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    encode_seconds = (
        time.perf_counter() - encode_started
    )

    dense_vectors = output["dense_vecs"]
    lexical_weights = output["lexical_weights"]

    if len(dense_vectors) != len(chunks):
        raise RuntimeError(
            "Dense output count does not match input chunks."
        )

    if len(lexical_weights) != len(chunks):
        raise RuntimeError(
            "Sparse output count does not match input chunks."
        )

    records: list[ChunkEmbeddingRecord] = []

    for index, chunk in enumerate(chunks):
        dense_vector = tuple(
            float(value)
            for value in dense_vectors[index]
        )

        sparse_vector = convert_sparse_vector(
            lexical_weights[index]
        )

        records.append(
            ChunkEmbeddingRecord(
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                doc_id=chunk.doc_id,
                dataset=chunk.dataset,
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_id=configuration[
                    "tokenizer_id"
                ],
                embedding_config_hash=config_hash,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                dense_dimension=len(dense_vector),
                dense_precision=(
                    VectorPrecision.FLOAT32
                ),
                dense_normalized=True,
            )
        )

    shard_id = build_embedding_shard_id(
        dataset="clerk",
        model_id=model_id,
        embedding_config_hash=config_hash,
        shard_index=0,
    )

    header = EmbeddingShardHeader(
        shard_id=shard_id,
        shard_index=0,
        dataset="clerk",
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=configuration[
            "tokenizer_id"
        ],
        embedding_config_hash=config_hash,
        embedding_kind=EmbeddingKind.HYBRID,
        dense_dimension=DENSE_DIMENSION,
        dense_precision=VectorPrecision.FLOAT32,
        dense_normalized=True,
        maximum_records=SMOKE_LIMIT,
    )

    summary = write_embedding_shard(
        root=SHARD_ROOT,
        header=header,
        records=records,
        source_chunks_path=CHUNKS_PATH,
        source_chunks_manifest_path=(
            CHUNKS_MANIFEST_PATH
        ),
        overwrite=True,
        execution_profile="local_cpu",
        metadata={
            "synthetic": False,
            "semantic_embeddings": True,
            "smoke_test": True,
            "model_load_seconds": round(
                load_seconds,
                3,
            ),
            "encode_seconds": round(
                encode_seconds,
                3,
            ),
        },
    )

    print("[OK] Model load seconds:", round(load_seconds, 3))
    print("[OK] Encode seconds:", round(encode_seconds, 3))
    print("[OK] Records:", summary.record_count)
    print("[OK] Dense dimension:", len(records[0].dense_vector or ()))
    print(
        "[OK] Sparse non-zero ranges:",
        min(
            len(record.sparse_vector.indices)
            for record in records
            if record.sparse_vector is not None
        ),
        max(
            len(record.sparse_vector.indices)
            for record in records
            if record.sparse_vector is not None
        ),
    )
    print("[OK] Shard ID:", summary.shard_id)
    print("[OK] Manifest:", summary.paths.manifest.resolve())


if __name__ == "__main__":
    main()
